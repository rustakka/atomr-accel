//! `PermutationRequest<T>` — dtype-generic mode permutation.
//!
//! B(modes_b) = alpha * op_A(A(modes_a)). Mirrors
//! `cutensorCreatePermutation` + `cutensorPermute`.

use std::sync::Arc;

use cudarc::cutensor::result as ct_result;
use cudarc::cutensor::sys as ct_sys;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::dtype::TensorSupported;
use crate::error::GpuError;
use crate::kernel::dispatch::{TensorDispatch, TensorDispatchCtx};
use crate::kernel::envelope;
use crate::kernel::tensor::compute_desc::{compute_desc_tag, resolve_compute_desc, ComputeDesc};
use crate::kernel::tensor::contract::OperandSpec;
use crate::kernel::tensor::plan_cache::{
    hash_i32s, hash_i64s, CachedPlan, OpKind, PlanCache, PlanKey,
};
use crate::kernel::tensor::SendHandle;
use crate::sys::cutensor as ct_local;

const LIB: &str = "cutensor";

pub struct PermutationRequest<T: TensorSupported> {
    pub a: OperandSpec<T>,
    pub b: OperandSpec<T>,
    pub alpha: T,
    pub op_a: ct_sys::cutensorOperator_t,
    pub compute: ComputeDesc,
    pub alignment: u32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: TensorSupported> PermutationRequest<T> {
    pub fn new(
        a: OperandSpec<T>,
        b: OperandSpec<T>,
        alpha: T,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            a,
            b,
            alpha,
            op_a: ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
            compute: super::contract::default_compute_for::<T>(),
            alignment: 16,
            reply,
        }
    }
}

impl<T: TensorSupported> TensorDispatch for PermutationRequest<T> {
    fn op_tag(&self) -> &'static str {
        "permute"
    }
    fn dtype_tag(&self) -> &'static str {
        <T as atomr_accel::AccelDtype>::NAME
    }
    fn dispatch(self: Box<Self>, ctx: &TensorDispatchCtx) {
        execute(*self, ctx);
    }
    fn fail_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "TensorActor in mock mode".into(),
        )));
    }
}

fn execute<T: TensorSupported>(req: PermutationRequest<T>, ctx: &TensorDispatchCtx) {
    let PermutationRequest {
        a,
        b,
        alpha,
        op_a,
        compute,
        alignment,
        reply,
    } = req;

    if a.extent.len() != a.modes.len() || b.extent.len() != b.modes.len() {
        let _ = reply.send(Err(GpuError::Unrecoverable(
            "Permutation: extent/modes length mismatch".into(),
        )));
        return;
    }

    let a_slice = match a.buf.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let b_slice = match b.buf.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut b_owned = match Arc::try_unwrap(b_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Permutation b has multiple live references".into(),
            )));
            return;
        }
    };

    let key = build_permutation_key_raw(
        <T as atomr_accel::AccelDtype>::NAME,
        &a.modes,
        &b.modes,
        &a.extent,
        &b.extent,
        alignment,
        compute,
        op_a,
    );
    let cached = match get_or_build_plan::<T>(
        &ctx.handle,
        &ctx.plan_cache,
        &key,
        &a,
        &b,
        alignment,
        compute,
        op_a,
    ) {
        Ok(p) => p,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    b.buf.record_write(&ctx.stream);

    let stream_for_check = ctx.stream.clone();
    let handle_clone = ctx.handle.clone();
    let plan_keepalive = cached.clone();

    envelope::run_kernel(LIB, &ctx.stream, &ctx.completion, (), reply, move || {
        let h = handle_clone.lock();
        let (a_ptr, _ga) = a_slice.device_ptr(&stream_for_check);
        let (b_ptr, _gb) = b_owned.device_ptr_mut(&stream_for_check);
        let alpha_h = alpha;
        let res = unsafe {
            ct_local::permute(
                h.0,
                plan_keepalive.plan,
                &alpha_h as *const T as *const _,
                a_ptr as *const _,
                b_ptr as *mut _,
                stream_for_check.cu_stream() as *mut _,
            )
        };
        drop((_ga, _gb));
        match res {
            Ok(()) => Ok((b_owned, a_slice, plan_keepalive)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Permute: {e}"),
            }),
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub fn build_permutation_key_raw(
    dtype_tag: &'static str,
    modes_a: &[i32],
    modes_b: &[i32],
    extent_a: &[i64],
    extent_b: &[i64],
    alignment: u32,
    compute: ComputeDesc,
    op_a: ct_sys::cutensorOperator_t,
) -> PlanKey {
    let mut modes = Vec::with_capacity(modes_a.len() + modes_b.len() + 2);
    modes.extend_from_slice(modes_a);
    modes.push(i32::MIN);
    modes.extend_from_slice(modes_b);
    let mut extents = Vec::with_capacity(extent_a.len() + extent_b.len() + 2);
    extents.extend_from_slice(extent_a);
    extents.push(i64::MIN);
    extents.extend_from_slice(extent_b);
    PlanKey {
        op_kind: OpKind::Permutation,
        modes_hash: hash_i32s(&modes),
        extents_hash: hash_i64s(&extents),
        alignment,
        compute_desc_tag: compute_desc_tag(compute) ^ ((op_a as u32).wrapping_mul(0x27d4_eb2f)),
        dtype_tag,
        algo: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn get_or_build_plan<T: TensorSupported>(
    handle: &Arc<Mutex<SendHandle>>,
    plan_cache: &Arc<PlanCache>,
    key: &PlanKey,
    a: &OperandSpec<T>,
    b: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    op_a: ct_sys::cutensorOperator_t,
) -> Result<Arc<CachedPlan>, GpuError> {
    if let Some(p) = plan_cache.get(key) {
        return Ok(p);
    }
    let plan = build_plan::<T>(handle, a, b, alignment, compute, op_a)?;
    let arc = Arc::new(plan);
    plan_cache.put(*key, arc.clone());
    Ok(arc)
}

fn build_plan<T: TensorSupported>(
    handle: &Arc<Mutex<SendHandle>>,
    a: &OperandSpec<T>,
    b: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    op_a: ct_sys::cutensorOperator_t,
) -> Result<CachedPlan, GpuError> {
    let h = handle.lock();
    let dt: cudarc::cutensor::sys::cudaDataType_t = unsafe { std::mem::transmute(T::cuda_data_type() as u32) };
    let cd = resolve_compute_desc(compute);
    let stride_ptr = |v: &Vec<i64>| {
        if v.is_empty() {
            std::ptr::null()
        } else {
            v.as_ptr()
        }
    };
    let desc_a = unsafe {
        ct_result::create_tensor_descriptor(
            h.0,
            a.extent.len() as u32,
            a.extent.as_ptr(),
            stride_ptr(&a.stride),
            dt,
            alignment,
        )
    }
    .map_err(|e| GpuError::lib(LIB, format!("CreateTensorDescriptor(A): {e}")))?;
    let desc_b = unsafe {
        ct_result::create_tensor_descriptor(
            h.0,
            b.extent.len() as u32,
            b.extent.as_ptr(),
            stride_ptr(&b.stride),
            dt,
            alignment,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreateTensorDescriptor(B): {e}"))
    })?;

    let op = unsafe {
        ct_local::create_permutation(
            h.0,
            desc_a,
            a.modes.as_ptr(),
            op_a,
            desc_b,
            b.modes.as_ptr(),
            cd,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreatePermutation: {e}"))
    })?;

    let pref = unsafe {
        ct_result::create_plan_preference(
            h.0,
            ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
            ct_sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_operation_descriptor(op);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreatePlanPreference: {e}"))
    })?;

    let ws_size = unsafe {
        ct_result::estimate_workspace_size(
            h.0,
            op,
            pref,
            ct_sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_DEFAULT,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_plan_preference(pref);
            let _ = ct_result::destroy_operation_descriptor(op);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("EstimateWorkspaceSize: {e}"))
    })?;

    let plan = unsafe { ct_result::create_plan(h.0, op, pref, ws_size) }.map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_plan_preference(pref);
            let _ = ct_result::destroy_operation_descriptor(op);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreatePlan: {e}"))
    })?;

    Ok(CachedPlan {
        plan,
        pref,
        op,
        descs: vec![desc_a, desc_b],
        workspace_size: ws_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_request_round_trip() {
        let key32 = build_permutation_key_raw(
            <f32 as atomr_accel::AccelDtype>::NAME,
            &[1, 2, 3],
            &[3, 1, 2],
            &[8, 16, 32],
            &[32, 8, 16],
            16,
            ComputeDesc::MinF32,
            ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
        );
        let key32_neg = build_permutation_key_raw(
            <f32 as atomr_accel::AccelDtype>::NAME,
            &[1, 2, 3],
            &[3, 1, 2],
            &[8, 16, 32],
            &[32, 8, 16],
            16,
            ComputeDesc::MinF32,
            ct_sys::cutensorOperator_t::CUTENSOR_OP_NEG,
        );
        // Same shapes, different op_a → distinct.
        assert_ne!(key32, key32_neg);
        assert_eq!(key32.op_kind, OpKind::Permutation);
        assert_eq!(key32.dtype_tag, "f32");

        let key64 = build_permutation_key_raw(
            <f64 as atomr_accel::AccelDtype>::NAME,
            &[1, 2, 3],
            &[3, 1, 2],
            &[8, 16, 32],
            &[32, 8, 16],
            16,
            ComputeDesc::MinF64,
            ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
        );
        assert_ne!(key32, key64);
        assert_eq!(key64.dtype_tag, "f64");
    }
}
