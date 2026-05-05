//! `ContractRequest<T>` — dtype-generic Einstein-summation contraction.
//!
//! D = alpha * A^modes_a · B^modes_b + beta * C^modes_c (D in-place
//! into the C buffer). Mirrors the cuTENSOR `cutensorContract` entry
//! point.

use std::sync::Arc;

use cudarc::cutensor::result as ct_result;
use cudarc::cutensor::sys as ct_sys;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use tokio::sync::oneshot;

use crate::dtype::TensorSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{TensorDispatch, TensorDispatchCtx};
use crate::kernel::envelope;
use crate::kernel::tensor::compute_desc::{compute_desc_tag, resolve_compute_desc, ComputeDesc};
use crate::kernel::tensor::plan_cache::{hash_i32s, hash_i64s, CachedPlan, OpKind, PlanKey};

const LIB: &str = "cutensor";

/// One operand specification: device buffer + per-mode extents +
/// optional strides + Einstein-summation labels.
#[derive(Clone)]
pub struct OperandSpec<T: TensorSupported> {
    pub buf: GpuRef<T>,
    pub extent: Vec<i64>,
    /// Empty == dense column-major.
    pub stride: Vec<i64>,
    pub modes: Vec<i32>,
}

/// Dtype-generic contraction request.
pub struct ContractRequest<T: TensorSupported> {
    pub a: OperandSpec<T>,
    pub b: OperandSpec<T>,
    pub c: OperandSpec<T>,
    pub alpha: T,
    pub beta: T,
    pub compute: ComputeDesc,
    /// Required tensor alignment in bytes.
    pub alignment: u32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: TensorSupported> ContractRequest<T> {
    pub fn new(
        a: OperandSpec<T>,
        b: OperandSpec<T>,
        c: OperandSpec<T>,
        alpha: T,
        beta: T,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            a,
            b,
            c,
            alpha,
            beta,
            compute: default_compute_for::<T>(),
            alignment: 16,
            reply,
        }
    }

    pub fn with_compute(mut self, compute: ComputeDesc) -> Self {
        self.compute = compute;
        self
    }
}

/// Pick the canonical compute descriptor for `T`. Mirrors NVIDIA's
/// guidance: f32 inputs default to MIN_32F, f64 to MIN_64F, half/bf16
/// accumulate in f32 (MIN_32F).
pub fn default_compute_for<T: TensorSupported>() -> ComputeDesc {
    match <T as atomr_accel::AccelDtype>::NAME {
        "f32" => ComputeDesc::MinF32,
        "f64" => ComputeDesc::MinF64,
        "f16" => ComputeDesc::MinF32,
        "bf16" => ComputeDesc::MinF32,
        _ => ComputeDesc::MinF32,
    }
}

impl<T: TensorSupported> TensorDispatch for ContractRequest<T> {
    fn op_tag(&self) -> &'static str {
        "contract"
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

fn execute<T: TensorSupported>(req: ContractRequest<T>, ctx: &TensorDispatchCtx) {
    let ContractRequest {
        a,
        b,
        c,
        alpha,
        beta,
        compute,
        alignment,
        reply,
    } = req;

    if a.extent.len() != a.modes.len() {
        let _ = reply.send(Err(GpuError::Unrecoverable(
            "Contract: a.extent.len != a.modes.len".into(),
        )));
        return;
    }
    if b.extent.len() != b.modes.len() {
        let _ = reply.send(Err(GpuError::Unrecoverable(
            "Contract: b.extent.len != b.modes.len".into(),
        )));
        return;
    }
    if c.extent.len() != c.modes.len() {
        let _ = reply.send(Err(GpuError::Unrecoverable(
            "Contract: c.extent.len != c.modes.len".into(),
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
                "Contract c has multiple live references".into(),
            )));
            return;
        }
    };

    let key = build_key_for::<T>(&a, &b, &c, alignment, compute, /*algo*/ 0);
    let cached = match get_or_build_plan::<T>(ctx, &key, &a, &b, &c, alignment, compute, None) {
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
        let (b_ptr, _gb) = b_slice.device_ptr(&stream_for_check);
        let (c_ptr, _gc) = c_owned.device_ptr_mut(&stream_for_check);
        let alpha_h = alpha;
        let beta_h = beta;
        let res = workspace
            .with_bucket(ws_size, |ws_slice| {
                let (ws_ptr, _gws) = ws_slice.device_ptr_mut(&stream_for_check);
                let r = unsafe {
                    ct_result::contract(
                        h.0,
                        plan_keepalive.plan,
                        &alpha_h as *const T as *const _,
                        a_ptr as *const _,
                        b_ptr as *const _,
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
                ct_result::contract(
                    h.0,
                    plan_keepalive.plan,
                    &alpha_h as *const T as *const _,
                    a_ptr as *const _,
                    b_ptr as *const _,
                    &beta_h as *const T as *const _,
                    c_ptr as *const _,
                    c_ptr as *mut _,
                    std::ptr::null_mut(),
                    0,
                    stream_for_check.cu_stream() as *mut _,
                )
            });
        drop((_ga, _gb, _gc));
        match res {
            Ok(()) => Ok((c_owned, a_slice, b_slice, plan_keepalive)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Contract: {e}"),
            }),
        }
    });
}

/// Build a cache key from raw mode/extent slices. Tests that don't
/// need a live `GpuRef` use this directly; the dispatch path goes
/// through [`build_key_for`].
pub fn build_contract_key(
    dtype_tag: &'static str,
    modes_a: &[i32],
    modes_b: &[i32],
    modes_c: &[i32],
    extent_a: &[i64],
    extent_b: &[i64],
    extent_c: &[i64],
    alignment: u32,
    compute: ComputeDesc,
    algo: i32,
) -> PlanKey {
    let mut modes = Vec::with_capacity(modes_a.len() + modes_b.len() + modes_c.len() + 3);
    modes.extend_from_slice(modes_a);
    modes.push(i32::MIN);
    modes.extend_from_slice(modes_b);
    modes.push(i32::MIN);
    modes.extend_from_slice(modes_c);
    let mut extents = Vec::with_capacity(extent_a.len() + extent_b.len() + extent_c.len() + 3);
    extents.extend_from_slice(extent_a);
    extents.push(i64::MIN);
    extents.extend_from_slice(extent_b);
    extents.push(i64::MIN);
    extents.extend_from_slice(extent_c);
    PlanKey {
        op_kind: OpKind::Contract,
        modes_hash: hash_i32s(&modes),
        extents_hash: hash_i64s(&extents),
        alignment,
        compute_desc_tag: compute_desc_tag(compute),
        dtype_tag,
        algo,
    }
}

/// Wrapper around [`build_contract_key`] that pulls extents/modes
/// from typed `OperandSpec`s.
pub(crate) fn build_key_for<T: TensorSupported>(
    a: &OperandSpec<T>,
    b: &OperandSpec<T>,
    c: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    algo: i32,
) -> PlanKey {
    build_contract_key(
        <T as atomr_accel::AccelDtype>::NAME,
        &a.modes,
        &b.modes,
        &c.modes,
        &a.extent,
        &b.extent,
        &c.extent,
        alignment,
        compute,
        algo,
    )
}

/// Look up `key` in `cache`; on miss, build a fresh plan with the
/// supplied algo (or default) and insert before returning.
pub(crate) fn get_or_build_plan<T: TensorSupported>(
    ctx: &TensorDispatchCtx,
    key: &PlanKey,
    a: &OperandSpec<T>,
    b: &OperandSpec<T>,
    c: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    algo: Option<ct_sys::cutensorAlgo_t>,
) -> Result<Arc<CachedPlan>, GpuError> {
    if let Some(p) = ctx.plan_cache.get(key) {
        return Ok(p);
    }
    let plan = build_plan::<T>(&ctx.handle, a, b, c, alignment, compute, algo)?;
    let arc = Arc::new(plan);
    ctx.plan_cache.put(*key, arc.clone());
    Ok(arc)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_plan<T: TensorSupported>(
    handle: &Arc<parking_lot::Mutex<crate::kernel::tensor::SendHandle>>,
    a: &OperandSpec<T>,
    b: &OperandSpec<T>,
    c: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    algo: Option<ct_sys::cutensorAlgo_t>,
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
    .map_err(|e| GpuError::LibraryError {
        lib: LIB,
        msg: format!("CreateTensorDescriptor(A): {e}"),
    })?;
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
        GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreateTensorDescriptor(B): {e}"),
        }
    })?;
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
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreateTensorDescriptor(C): {e}"),
        }
    })?;

    let op = unsafe {
        ct_result::create_contraction(
            h.0,
            desc_a,
            a.modes.as_ptr(),
            ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
            desc_b,
            b.modes.as_ptr(),
            ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
            desc_c,
            c.modes.as_ptr(),
            ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
            desc_c,
            c.modes.as_ptr(),
            cd,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreateContraction: {e}"),
        }
    })?;

    let chosen_algo = algo.unwrap_or(ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT);
    let pref = unsafe {
        ct_result::create_plan_preference(h.0, chosen_algo, ct_sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE)
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_operation_descriptor(op);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreatePlanPreference: {e}"),
        }
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
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::LibraryError {
            lib: LIB,
            msg: format!("EstimateWorkspaceSize: {e}"),
        }
    })?;

    let plan = unsafe { ct_result::create_plan(h.0, op, pref, ws_size) }.map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_plan_preference(pref);
            let _ = ct_result::destroy_operation_descriptor(op);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreatePlan: {e}"),
        }
    })?;

    Ok(CachedPlan {
        plan,
        pref,
        op,
        descs: vec![desc_a, desc_b, desc_c],
        workspace_size: ws_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_request_round_trip_f32_f64_f16_bf16() {
        // Round-trip: build cache keys for each supported dtype and
        // verify dtype tag, op kind, and compute descriptor are wired
        // correctly. Cache keys are derived from raw mode/extent
        // vectors, so this test runs on a GPU-less host.
        let key32 = build_contract_key(
            <f32 as atomr_accel::AccelDtype>::NAME,
            &[1, 2],
            &[2, 3],
            &[1, 3],
            &[2, 3],
            &[3, 4],
            &[2, 4],
            16,
            ComputeDesc::MinF32,
            0,
        );
        assert_eq!(key32.dtype_tag, "f32");
        assert_eq!(key32.op_kind, OpKind::Contract);
        assert_eq!(key32.compute_desc_tag, compute_desc_tag(ComputeDesc::MinF32));

        let key64 = build_contract_key(
            <f64 as atomr_accel::AccelDtype>::NAME,
            &[1, 2],
            &[2, 3],
            &[1, 3],
            &[2, 3],
            &[3, 4],
            &[2, 4],
            16,
            ComputeDesc::MinF64,
            0,
        );
        assert_eq!(key64.dtype_tag, "f64");
        // Different dtype must produce a different key even with the
        // same shapes.
        assert_ne!(key32, key64);

        // Default compute descriptor per dtype.
        assert_eq!(default_compute_for::<f32>().tag(), ComputeDesc::MinF32.tag());
        assert_eq!(default_compute_for::<f64>().tag(), ComputeDesc::MinF64.tag());

        #[cfg(feature = "f16")]
        {
            let key_f16 = build_contract_key(
                <half::f16 as atomr_accel::AccelDtype>::NAME,
                &[1, 2],
                &[2, 3],
                &[1, 3],
                &[2, 3],
                &[3, 4],
                &[2, 4],
                16,
                ComputeDesc::MinF32,
                0,
            );
            assert_eq!(key_f16.dtype_tag, "f16");
            assert_ne!(key32, key_f16);

            let key_bf16 = build_contract_key(
                <half::bf16 as atomr_accel::AccelDtype>::NAME,
                &[1, 2],
                &[2, 3],
                &[1, 3],
                &[2, 3],
                &[3, 4],
                &[2, 4],
                16,
                ComputeDesc::MinF32,
                0,
            );
            assert_eq!(key_bf16.dtype_tag, "bf16");
            assert_ne!(key_f16, key_bf16);

            assert_eq!(
                default_compute_for::<half::f16>().tag(),
                ComputeDesc::MinF32.tag()
            );
            assert_eq!(
                default_compute_for::<half::bf16>().tag(),
                ComputeDesc::MinF32.tag()
            );
        }
    }
}
