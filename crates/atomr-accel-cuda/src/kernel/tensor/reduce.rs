//! `ReductionRequest<T>` — dtype-generic tensor reduction.
//!
//! Wraps `cutensorCreateReduction` + `cutensorReduce`. Output `c`
//! holds the reduced result (shape determined by `mode_c` ⊆
//! `mode_a`). `op_reduce` is one of `CUTENSOR_OP_ADD`,
//! `CUTENSOR_OP_MAX`, `CUTENSOR_OP_MIN`, `CUTENSOR_OP_MUL`.

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
    hash_i32s, hash_i64s, CachedPlan, OpKind, PlanKey,
};
use crate::kernel::tensor::SendHandle;
use crate::sys::cutensor as ct_local;

const LIB: &str = "cutensor";

/// Dtype-generic reduction request.
pub struct ReductionRequest<T: TensorSupported> {
    pub a: OperandSpec<T>,
    pub c: OperandSpec<T>,
    pub alpha: T,
    pub beta: T,
    pub op_reduce: ct_sys::cutensorOperator_t,
    pub compute: ComputeDesc,
    pub alignment: u32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: TensorSupported> ReductionRequest<T> {
    pub fn new(
        a: OperandSpec<T>,
        c: OperandSpec<T>,
        alpha: T,
        beta: T,
        op_reduce: ct_sys::cutensorOperator_t,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            a,
            c,
            alpha,
            beta,
            op_reduce,
            compute: super::contract::default_compute_for::<T>(),
            alignment: 16,
            reply,
        }
    }
}

impl<T: TensorSupported> TensorDispatch for ReductionRequest<T> {
    fn op_tag(&self) -> &'static str {
        "reduce"
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

fn execute<T: TensorSupported>(req: ReductionRequest<T>, ctx: &TensorDispatchCtx) {
    let ReductionRequest {
        a,
        c,
        alpha,
        beta,
        op_reduce,
        compute,
        alignment,
        reply,
    } = req;

    if a.extent.len() != a.modes.len() {
        let _ = reply.send(Err(GpuError::Unrecoverable(
            "Reduce: a.extent.len != a.modes.len".into(),
        )));
        return;
    }
    if c.extent.len() != c.modes.len() {
        let _ = reply.send(Err(GpuError::Unrecoverable(
            "Reduce: c.extent.len != c.modes.len".into(),
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
    let c_slice = match c.buf.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut c_owned = match Arc::try_unwrap(c_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Reduce c has multiple live references".into(),
            )));
            return;
        }
    };

    let key = build_reduction_key::<T>(&a, &c, alignment, compute, op_reduce);
    let cached = match get_or_build_plan::<T>(&ctx.handle, &ctx.plan_cache, &key, &a, &c, alignment, compute, op_reduce) {
        Ok(p) => p,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    let ws_size = cached.workspace_size as usize;
    if let Err(e) = ctx.workspace.ensure(ws_size) {
        let _ = reply.send(Err(e));
        return;
    }

    c.buf.record_write(&ctx.stream);

    let stream_for_check = ctx.stream.clone();
    let handle_clone = ctx.handle.clone();
    let workspace = ctx.workspace.clone();
    let plan_keepalive = cached.clone();

    envelope::run_kernel(LIB, &ctx.stream, &ctx.completion, (), reply, move || {
        let h = handle_clone.lock();
        let (a_ptr, _ga) = a_slice.device_ptr(&stream_for_check);
        let (c_ptr, _gc) = c_owned.device_ptr_mut(&stream_for_check);
        let alpha_h = alpha;
        let beta_h = beta;
        let res = workspace
            .with_bucket(ws_size, |ws_slice| {
                let (ws_ptr, _gws) = ws_slice.device_ptr_mut(&stream_for_check);
                let r = unsafe {
                    ct_local::reduce(
                        h.0,
                        plan_keepalive.plan,
                        &alpha_h as *const T as *const _,
                        a_ptr as *const _,
                        &beta_h as *const T as *const _,
                        c_ptr as *const _,
                        c_ptr as *mut _,
                        ws_ptr as *mut _,
                        plan_keepalive.workspace_size,
                        stream_for_check.cu_stream() as *mut _,
                    )
                };
                drop(_gws);
                r
            })
            .unwrap_or_else(|| unsafe {
                ct_local::reduce(
                    h.0,
                    plan_keepalive.plan,
                    &alpha_h as *const T as *const _,
                    a_ptr as *const _,
                    &beta_h as *const T as *const _,
                    c_ptr as *const _,
                    c_ptr as *mut _,
                    std::ptr::null_mut(),
                    0,
                    stream_for_check.cu_stream() as *mut _,
                )
            });
        drop((_ga, _gc));
        match res {
            Ok(()) => Ok((c_owned, a_slice, plan_keepalive)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Reduce: {e}"),
            }),
        }
    });
}

pub fn build_reduction_key_raw(
    dtype_tag: &'static str,
    modes_a: &[i32],
    modes_c: &[i32],
    extent_a: &[i64],
    extent_c: &[i64],
    alignment: u32,
    compute: ComputeDesc,
    op_reduce: ct_sys::cutensorOperator_t,
) -> PlanKey {
    let mut modes = Vec::with_capacity(modes_a.len() + modes_c.len() + 2);
    modes.extend_from_slice(modes_a);
    modes.push(i32::MIN);
    modes.extend_from_slice(modes_c);
    let mut extents = Vec::with_capacity(extent_a.len() + extent_c.len() + 2);
    extents.extend_from_slice(extent_a);
    extents.push(i64::MIN);
    extents.extend_from_slice(extent_c);
    PlanKey {
        op_kind: OpKind::Reduce,
        modes_hash: hash_i32s(&modes),
        extents_hash: hash_i64s(&extents),
        alignment,
        // Fold the reduction operator into the compute-desc tag so two
        // shapes-only-equal-but-op-different requests don't collide.
        compute_desc_tag: compute_desc_tag(compute) ^ ((op_reduce as u32).wrapping_mul(0x9E37_79B9)),
        dtype_tag,
        algo: 0,
    }
}

fn build_reduction_key<T: TensorSupported>(
    a: &OperandSpec<T>,
    c: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    op_reduce: ct_sys::cutensorOperator_t,
) -> PlanKey {
    build_reduction_key_raw(
        <T as atomr_accel::AccelDtype>::NAME,
        &a.modes,
        &c.modes,
        &a.extent,
        &c.extent,
        alignment,
        compute,
        op_reduce,
    )
}

#[allow(clippy::too_many_arguments)]
fn get_or_build_plan<T: TensorSupported>(
    handle: &Arc<Mutex<SendHandle>>,
    plan_cache: &Arc<crate::kernel::tensor::plan_cache::PlanCache>,
    key: &PlanKey,
    a: &OperandSpec<T>,
    c: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    op_reduce: ct_sys::cutensorOperator_t,
) -> Result<Arc<CachedPlan>, GpuError> {
    if let Some(p) = plan_cache.get(key) {
        return Ok(p);
    }
    let plan = build_plan::<T>(handle, a, c, alignment, compute, op_reduce)?;
    let arc = Arc::new(plan);
    plan_cache.put(*key, arc.clone());
    Ok(arc)
}

fn build_plan<T: TensorSupported>(
    handle: &Arc<Mutex<SendHandle>>,
    a: &OperandSpec<T>,
    c: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    op_reduce: ct_sys::cutensorOperator_t,
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
    let desc_c = unsafe {
        ct_result::create_tensor_descriptor(
            h.0,
            c.extent.len() as u32,
            c.extent.as_ptr(),
            stride_ptr(&c.stride),
            dt,
            alignment,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreateTensorDescriptor(C): {e}"))
    })?;

    let op = unsafe {
        ct_result::create_reduction(
            h.0,
            desc_a,
            a.modes.as_ptr(),
            ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
            desc_c,
            c.modes.as_ptr(),
            ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
            desc_c,
            c.modes.as_ptr(),
            op_reduce,
            cd,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreateReduction: {e}"))
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
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
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
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("EstimateWorkspaceSize: {e}"))
    })?;

    let plan = unsafe { ct_result::create_plan(h.0, op, pref, ws_size) }.map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_plan_preference(pref);
            let _ = ct_result::destroy_operation_descriptor(op);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreatePlan: {e}"))
    })?;

    Ok(CachedPlan {
        plan,
        pref,
        op,
        descs: vec![desc_a, desc_c],
        workspace_size: ws_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_request_round_trip() {
        let key_add = build_reduction_key_raw(
            <f32 as atomr_accel::AccelDtype>::NAME,
            &[1, 2, 3],
            &[1],
            &[8, 16, 32],
            &[8],
            16,
            ComputeDesc::MinF32,
            ct_sys::cutensorOperator_t::CUTENSOR_OP_ADD,
        );
        let key_max = build_reduction_key_raw(
            <f32 as atomr_accel::AccelDtype>::NAME,
            &[1, 2, 3],
            &[1],
            &[8, 16, 32],
            &[8],
            16,
            ComputeDesc::MinF32,
            ct_sys::cutensorOperator_t::CUTENSOR_OP_MAX,
        );
        // Same shapes, different op_reduce → distinct cache slots.
        assert_ne!(key_add, key_max);
        assert_eq!(key_add.op_kind, OpKind::Reduce);
        assert_eq!(key_add.dtype_tag, "f32");

        let key_f64 = build_reduction_key_raw(
            <f64 as atomr_accel::AccelDtype>::NAME,
            &[1, 2, 3],
            &[1],
            &[8, 16, 32],
            &[8],
            16,
            ComputeDesc::MinF64,
            ct_sys::cutensorOperator_t::CUTENSOR_OP_ADD,
        );
        assert_ne!(key_add, key_f64);
        assert_eq!(key_f64.dtype_tag, "f64");
    }
}
