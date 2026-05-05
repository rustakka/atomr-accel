//! `ElementwiseBinaryRequest<T>` and `ElementwiseTrinaryRequest<T>`.
//!
//! Both wrap the cuTENSOR `cutensorCreate{Binary,Trinary}` +
//! `cutensorElementwise{Binary,Trinary}Execute` pair via our local
//! `crate::sys::cutensor` wrappers — cudarc's safe layer doesn't
//! expose these.

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

// ---------------------------------------------------------------------
// Binary
// ---------------------------------------------------------------------

/// D = op_AC( alpha * op_A(A), gamma * op_C(C) ).
pub struct ElementwiseBinaryRequest<T: TensorSupported> {
    pub a: OperandSpec<T>,
    pub c: OperandSpec<T>,
    pub d: OperandSpec<T>,
    pub alpha: T,
    pub gamma: T,
    pub op_a: ct_sys::cutensorOperator_t,
    pub op_c: ct_sys::cutensorOperator_t,
    pub op_ac: ct_sys::cutensorOperator_t,
    pub compute: ComputeDesc,
    pub alignment: u32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: TensorSupported> ElementwiseBinaryRequest<T> {
    pub fn new(
        a: OperandSpec<T>,
        c: OperandSpec<T>,
        d: OperandSpec<T>,
        alpha: T,
        gamma: T,
        op_ac: ct_sys::cutensorOperator_t,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            a,
            c,
            d,
            alpha,
            gamma,
            op_a: ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
            op_c: ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
            op_ac,
            compute: super::contract::default_compute_for::<T>(),
            alignment: 16,
            reply,
        }
    }
}

impl<T: TensorSupported> TensorDispatch for ElementwiseBinaryRequest<T> {
    fn op_tag(&self) -> &'static str {
        "ewbin"
    }
    fn dtype_tag(&self) -> &'static str {
        T::dtype_tag()
    }
    fn dispatch(self: Box<Self>, ctx: &TensorDispatchCtx) {
        execute_binary(*self, ctx);
    }
    fn fail_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "TensorActor in mock mode".into(),
        )));
    }
}

fn execute_binary<T: TensorSupported>(req: ElementwiseBinaryRequest<T>, ctx: &TensorDispatchCtx) {
    let ElementwiseBinaryRequest {
        a,
        c,
        d,
        alpha,
        gamma,
        op_a,
        op_c,
        op_ac,
        compute,
        alignment,
        reply,
    } = req;

    if a.extent.len() != a.modes.len()
        || c.extent.len() != c.modes.len()
        || d.extent.len() != d.modes.len()
    {
        let _ = reply.send(Err(GpuError::Unrecoverable(
            "ElementwiseBinary: extent/modes length mismatch".into(),
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
    let d_slice = match d.buf.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut d_owned = match Arc::try_unwrap(d_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "ElementwiseBinary d has multiple live references".into(),
            )));
            return;
        }
    };

    let key = build_binary_key_raw(
        T::dtype_tag(),
        &a.modes,
        &c.modes,
        &d.modes,
        &a.extent,
        &c.extent,
        &d.extent,
        alignment,
        compute,
        op_a,
        op_c,
        op_ac,
    );
    let cached = match get_or_build_binary::<T>(
        &ctx.handle,
        &ctx.plan_cache,
        &key,
        &a,
        &c,
        &d,
        alignment,
        compute,
        op_a,
        op_c,
        op_ac,
    ) {
        Ok(p) => p,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    d.buf.record_write(&ctx.stream);

    let stream_for_check = ctx.stream.clone();
    let handle_clone = ctx.handle.clone();
    let plan_keepalive = cached.clone();

    envelope::run_kernel(LIB, &ctx.stream, &ctx.completion, (), reply, move || {
        let h = handle_clone.lock();
        let (a_ptr, _ga) = a_slice.device_ptr(&stream_for_check);
        let (c_ptr, _gc) = c_slice.device_ptr(&stream_for_check);
        let (d_ptr, _gd) = d_owned.device_ptr_mut(&stream_for_check);
        let alpha_h = alpha;
        let gamma_h = gamma;
        let res = unsafe {
            ct_local::elementwise_binary_execute(
                h.0,
                plan_keepalive.plan,
                &alpha_h as *const T as *const _,
                a_ptr as *const _,
                &gamma_h as *const T as *const _,
                c_ptr as *const _,
                d_ptr as *mut _,
                stream_for_check.cu_stream() as *mut _,
            )
        };
        drop((_ga, _gc, _gd));
        match res {
            Ok(()) => Ok((d_owned, a_slice, c_slice, plan_keepalive)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("ElementwiseBinary: {e}"),
            }),
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub fn build_binary_key_raw(
    dtype_tag: &'static str,
    modes_a: &[i32],
    modes_c: &[i32],
    modes_d: &[i32],
    extent_a: &[i64],
    extent_c: &[i64],
    extent_d: &[i64],
    alignment: u32,
    compute: ComputeDesc,
    op_a: ct_sys::cutensorOperator_t,
    op_c: ct_sys::cutensorOperator_t,
    op_ac: ct_sys::cutensorOperator_t,
) -> PlanKey {
    let mut modes = Vec::with_capacity(modes_a.len() + modes_c.len() + modes_d.len() + 3);
    modes.extend_from_slice(modes_a);
    modes.push(i32::MIN);
    modes.extend_from_slice(modes_c);
    modes.push(i32::MIN);
    modes.extend_from_slice(modes_d);
    let mut extents = Vec::with_capacity(extent_a.len() + extent_c.len() + extent_d.len() + 3);
    extents.extend_from_slice(extent_a);
    extents.push(i64::MIN);
    extents.extend_from_slice(extent_c);
    extents.push(i64::MIN);
    extents.extend_from_slice(extent_d);
    let op_mix = ((op_a as u32).wrapping_mul(0x85eb_ca77))
        ^ ((op_c as u32).wrapping_mul(0xc2b2_ae3d))
        ^ ((op_ac as u32).wrapping_mul(0x27d4_eb2f));
    PlanKey {
        op_kind: OpKind::ElementwiseBinary,
        modes_hash: hash_i32s(&modes),
        extents_hash: hash_i64s(&extents),
        alignment,
        compute_desc_tag: compute_desc_tag(compute) ^ op_mix,
        dtype_tag,
        algo: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn get_or_build_binary<T: TensorSupported>(
    handle: &Arc<Mutex<SendHandle>>,
    plan_cache: &Arc<PlanCache>,
    key: &PlanKey,
    a: &OperandSpec<T>,
    c: &OperandSpec<T>,
    d: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    op_a: ct_sys::cutensorOperator_t,
    op_c: ct_sys::cutensorOperator_t,
    op_ac: ct_sys::cutensorOperator_t,
) -> Result<Arc<CachedPlan>, GpuError> {
    if let Some(p) = plan_cache.get(key) {
        return Ok(p);
    }
    let plan = build_binary_plan::<T>(handle, a, c, d, alignment, compute, op_a, op_c, op_ac)?;
    let arc = Arc::new(plan);
    plan_cache.put(*key, arc.clone());
    Ok(arc)
}

#[allow(clippy::too_many_arguments)]
fn build_binary_plan<T: TensorSupported>(
    handle: &Arc<Mutex<SendHandle>>,
    a: &OperandSpec<T>,
    c: &OperandSpec<T>,
    d: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    op_a: ct_sys::cutensorOperator_t,
    op_c: ct_sys::cutensorOperator_t,
    op_ac: ct_sys::cutensorOperator_t,
) -> Result<CachedPlan, GpuError> {
    let h = handle.lock();
    let dt = T::cuda_data_type();
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
    let desc_d = unsafe {
        ct_result::create_tensor_descriptor(
            h.0,
            d.extent.len() as u32,
            d.extent.as_ptr(),
            stride_ptr(&d.stride),
            dt,
            alignment,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreateTensorDescriptor(D): {e}"))
    })?;

    let op = unsafe {
        ct_local::create_elementwise_binary(
            h.0,
            desc_a,
            a.modes.as_ptr(),
            op_a,
            desc_c,
            c.modes.as_ptr(),
            op_c,
            desc_d,
            d.modes.as_ptr(),
            op_ac,
            cd,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_tensor_descriptor(desc_d);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreateElementwiseBinary: {e}"))
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
            let _ = ct_result::destroy_tensor_descriptor(desc_d);
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
            let _ = ct_result::destroy_tensor_descriptor(desc_d);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("EstimateWorkspaceSize: {e}"))
    })?;

    let plan = unsafe { ct_result::create_plan(h.0, op, pref, ws_size) }.map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_plan_preference(pref);
            let _ = ct_result::destroy_operation_descriptor(op);
            let _ = ct_result::destroy_tensor_descriptor(desc_d);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreatePlan: {e}"))
    })?;

    Ok(CachedPlan {
        plan,
        pref,
        op,
        descs: vec![desc_a, desc_c, desc_d],
        workspace_size: ws_size,
    })
}

// ---------------------------------------------------------------------
// Trinary
// ---------------------------------------------------------------------

/// D = op_ABC( op_AB(alpha * op_A(A), beta * op_B(B)), gamma * op_C(C) ).
pub struct ElementwiseTrinaryRequest<T: TensorSupported> {
    pub a: OperandSpec<T>,
    pub b: OperandSpec<T>,
    pub c: OperandSpec<T>,
    pub d: OperandSpec<T>,
    pub alpha: T,
    pub beta: T,
    pub gamma: T,
    pub op_a: ct_sys::cutensorOperator_t,
    pub op_b: ct_sys::cutensorOperator_t,
    pub op_c: ct_sys::cutensorOperator_t,
    pub op_ab: ct_sys::cutensorOperator_t,
    pub op_abc: ct_sys::cutensorOperator_t,
    pub compute: ComputeDesc,
    pub alignment: u32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T: TensorSupported> ElementwiseTrinaryRequest<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: OperandSpec<T>,
        b: OperandSpec<T>,
        c: OperandSpec<T>,
        d: OperandSpec<T>,
        alpha: T,
        beta: T,
        gamma: T,
        op_ab: ct_sys::cutensorOperator_t,
        op_abc: ct_sys::cutensorOperator_t,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        let id = ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY;
        Self {
            a,
            b,
            c,
            d,
            alpha,
            beta,
            gamma,
            op_a: id,
            op_b: id,
            op_c: id,
            op_ab,
            op_abc,
            compute: super::contract::default_compute_for::<T>(),
            alignment: 16,
            reply,
        }
    }
}

impl<T: TensorSupported> TensorDispatch for ElementwiseTrinaryRequest<T> {
    fn op_tag(&self) -> &'static str {
        "ewtri"
    }
    fn dtype_tag(&self) -> &'static str {
        T::dtype_tag()
    }
    fn dispatch(self: Box<Self>, ctx: &TensorDispatchCtx) {
        execute_trinary(*self, ctx);
    }
    fn fail_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "TensorActor in mock mode".into(),
        )));
    }
}

fn execute_trinary<T: TensorSupported>(req: ElementwiseTrinaryRequest<T>, ctx: &TensorDispatchCtx) {
    let ElementwiseTrinaryRequest {
        a,
        b,
        c,
        d,
        alpha,
        beta,
        gamma,
        op_a,
        op_b,
        op_c,
        op_ab,
        op_abc,
        compute,
        alignment,
        reply,
    } = req;

    if a.extent.len() != a.modes.len()
        || b.extent.len() != b.modes.len()
        || c.extent.len() != c.modes.len()
        || d.extent.len() != d.modes.len()
    {
        let _ = reply.send(Err(GpuError::Unrecoverable(
            "ElementwiseTrinary: extent/modes length mismatch".into(),
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
    let d_slice = match d.buf.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut d_owned = match Arc::try_unwrap(d_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "ElementwiseTrinary d has multiple live references".into(),
            )));
            return;
        }
    };

    let key = build_trinary_key_raw(
        T::dtype_tag(),
        &a.modes,
        &b.modes,
        &c.modes,
        &d.modes,
        &a.extent,
        &b.extent,
        &c.extent,
        &d.extent,
        alignment,
        compute,
        op_a,
        op_b,
        op_c,
        op_ab,
        op_abc,
    );
    let cached = match get_or_build_trinary::<T>(
        &ctx.handle,
        &ctx.plan_cache,
        &key,
        &a,
        &b,
        &c,
        &d,
        alignment,
        compute,
        op_a,
        op_b,
        op_c,
        op_ab,
        op_abc,
    ) {
        Ok(p) => p,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    d.buf.record_write(&ctx.stream);

    let stream_for_check = ctx.stream.clone();
    let handle_clone = ctx.handle.clone();
    let plan_keepalive = cached.clone();

    envelope::run_kernel(LIB, &ctx.stream, &ctx.completion, (), reply, move || {
        let h = handle_clone.lock();
        let (a_ptr, _ga) = a_slice.device_ptr(&stream_for_check);
        let (b_ptr, _gb) = b_slice.device_ptr(&stream_for_check);
        let (c_ptr, _gc) = c_slice.device_ptr(&stream_for_check);
        let (d_ptr, _gd) = d_owned.device_ptr_mut(&stream_for_check);
        let alpha_h = alpha;
        let beta_h = beta;
        let gamma_h = gamma;
        let res = unsafe {
            ct_local::elementwise_trinary_execute(
                h.0,
                plan_keepalive.plan,
                &alpha_h as *const T as *const _,
                a_ptr as *const _,
                &beta_h as *const T as *const _,
                b_ptr as *const _,
                &gamma_h as *const T as *const _,
                c_ptr as *const _,
                d_ptr as *mut _,
                stream_for_check.cu_stream() as *mut _,
            )
        };
        drop((_ga, _gb, _gc, _gd));
        match res {
            Ok(()) => Ok((d_owned, a_slice, b_slice, c_slice, plan_keepalive)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("ElementwiseTrinary: {e}"),
            }),
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub fn build_trinary_key_raw(
    dtype_tag: &'static str,
    modes_a: &[i32],
    modes_b: &[i32],
    modes_c: &[i32],
    modes_d: &[i32],
    extent_a: &[i64],
    extent_b: &[i64],
    extent_c: &[i64],
    extent_d: &[i64],
    alignment: u32,
    compute: ComputeDesc,
    op_a: ct_sys::cutensorOperator_t,
    op_b: ct_sys::cutensorOperator_t,
    op_c: ct_sys::cutensorOperator_t,
    op_ab: ct_sys::cutensorOperator_t,
    op_abc: ct_sys::cutensorOperator_t,
) -> PlanKey {
    let mut modes =
        Vec::with_capacity(modes_a.len() + modes_b.len() + modes_c.len() + modes_d.len() + 4);
    modes.extend_from_slice(modes_a);
    modes.push(i32::MIN);
    modes.extend_from_slice(modes_b);
    modes.push(i32::MIN);
    modes.extend_from_slice(modes_c);
    modes.push(i32::MIN);
    modes.extend_from_slice(modes_d);
    let mut extents = Vec::with_capacity(
        extent_a.len() + extent_b.len() + extent_c.len() + extent_d.len() + 4,
    );
    extents.extend_from_slice(extent_a);
    extents.push(i64::MIN);
    extents.extend_from_slice(extent_b);
    extents.push(i64::MIN);
    extents.extend_from_slice(extent_c);
    extents.push(i64::MIN);
    extents.extend_from_slice(extent_d);
    let op_mix = ((op_a as u32).wrapping_mul(0x85eb_ca77))
        ^ ((op_b as u32).wrapping_mul(0xc2b2_ae3d))
        ^ ((op_c as u32).wrapping_mul(0x27d4_eb2f))
        ^ ((op_ab as u32).wrapping_mul(0x9e37_79b9))
        ^ ((op_abc as u32).wrapping_mul(0x6a09_e667));
    PlanKey {
        op_kind: OpKind::ElementwiseTrinary,
        modes_hash: hash_i32s(&modes),
        extents_hash: hash_i64s(&extents),
        alignment,
        compute_desc_tag: compute_desc_tag(compute) ^ op_mix,
        dtype_tag,
        algo: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn get_or_build_trinary<T: TensorSupported>(
    handle: &Arc<Mutex<SendHandle>>,
    plan_cache: &Arc<PlanCache>,
    key: &PlanKey,
    a: &OperandSpec<T>,
    b: &OperandSpec<T>,
    c: &OperandSpec<T>,
    d: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    op_a: ct_sys::cutensorOperator_t,
    op_b: ct_sys::cutensorOperator_t,
    op_c: ct_sys::cutensorOperator_t,
    op_ab: ct_sys::cutensorOperator_t,
    op_abc: ct_sys::cutensorOperator_t,
) -> Result<Arc<CachedPlan>, GpuError> {
    if let Some(p) = plan_cache.get(key) {
        return Ok(p);
    }
    let plan = build_trinary_plan::<T>(
        handle, a, b, c, d, alignment, compute, op_a, op_b, op_c, op_ab, op_abc,
    )?;
    let arc = Arc::new(plan);
    plan_cache.put(*key, arc.clone());
    Ok(arc)
}

#[allow(clippy::too_many_arguments)]
fn build_trinary_plan<T: TensorSupported>(
    handle: &Arc<Mutex<SendHandle>>,
    a: &OperandSpec<T>,
    b: &OperandSpec<T>,
    c: &OperandSpec<T>,
    d: &OperandSpec<T>,
    alignment: u32,
    compute: ComputeDesc,
    op_a: ct_sys::cutensorOperator_t,
    op_b: ct_sys::cutensorOperator_t,
    op_c: ct_sys::cutensorOperator_t,
    op_ab: ct_sys::cutensorOperator_t,
    op_abc: ct_sys::cutensorOperator_t,
) -> Result<CachedPlan, GpuError> {
    let h = handle.lock();
    let dt = T::cuda_data_type();
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
        GpuError::lib(LIB, format!("CreateTensorDescriptor(C): {e}"))
    })?;
    let desc_d = unsafe {
        ct_result::create_tensor_descriptor(
            h.0,
            d.extent.len() as u32,
            d.extent.as_ptr(),
            stride_ptr(&d.stride),
            dt,
            alignment,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreateTensorDescriptor(D): {e}"))
    })?;

    let op = unsafe {
        ct_local::create_elementwise_trinary(
            h.0,
            desc_a,
            a.modes.as_ptr(),
            op_a,
            desc_b,
            b.modes.as_ptr(),
            op_b,
            desc_c,
            c.modes.as_ptr(),
            op_c,
            desc_d,
            d.modes.as_ptr(),
            op_ab,
            op_abc,
            cd,
        )
    }
    .map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_tensor_descriptor(desc_d);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreateElementwiseTrinary: {e}"))
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
            let _ = ct_result::destroy_tensor_descriptor(desc_d);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
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
            let _ = ct_result::destroy_tensor_descriptor(desc_d);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("EstimateWorkspaceSize: {e}"))
    })?;

    let plan = unsafe { ct_result::create_plan(h.0, op, pref, ws_size) }.map_err(|e| {
        unsafe {
            let _ = ct_result::destroy_plan_preference(pref);
            let _ = ct_result::destroy_operation_descriptor(op);
            let _ = ct_result::destroy_tensor_descriptor(desc_d);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        GpuError::lib(LIB, format!("CreatePlan: {e}"))
    })?;

    Ok(CachedPlan {
        plan,
        pref,
        op,
        descs: vec![desc_a, desc_b, desc_c, desc_d],
        workspace_size: ws_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trinary_binary_request_round_trip() {
        let id = ct_sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY;
        let add = ct_sys::cutensorOperator_t::CUTENSOR_OP_ADD;
        let mul = ct_sys::cutensorOperator_t::CUTENSOR_OP_MUL;

        // Binary
        let key_b1 = build_binary_key_raw(
            <f32 as TensorSupported>::dtype_tag(),
            &[1, 2],
            &[1, 2],
            &[1, 2],
            &[8, 16],
            &[8, 16],
            &[8, 16],
            16,
            ComputeDesc::MinF32,
            id,
            id,
            add,
        );
        let key_b2 = build_binary_key_raw(
            <f32 as TensorSupported>::dtype_tag(),
            &[1, 2],
            &[1, 2],
            &[1, 2],
            &[8, 16],
            &[8, 16],
            &[8, 16],
            16,
            ComputeDesc::MinF32,
            id,
            id,
            mul,
        );
        // Same shapes, different op_ac → distinct.
        assert_ne!(key_b1, key_b2);
        assert_eq!(key_b1.op_kind, OpKind::ElementwiseBinary);

        // Trinary
        let key_t1 = build_trinary_key_raw(
            <f64 as TensorSupported>::dtype_tag(),
            &[1, 2],
            &[1, 2],
            &[1, 2],
            &[1, 2],
            &[8, 16],
            &[8, 16],
            &[8, 16],
            &[8, 16],
            16,
            ComputeDesc::MinF64,
            id,
            id,
            id,
            add,
            mul,
        );
        let key_t2 = build_trinary_key_raw(
            <f64 as TensorSupported>::dtype_tag(),
            &[1, 2],
            &[1, 2],
            &[1, 2],
            &[1, 2],
            &[8, 16],
            &[8, 16],
            &[8, 16],
            &[8, 16],
            16,
            ComputeDesc::MinF64,
            id,
            id,
            id,
            mul,
            add,
        );
        assert_ne!(key_t1, key_t2);
        assert_eq!(key_t1.op_kind, OpKind::ElementwiseTrinary);
        assert_eq!(key_t1.dtype_tag, "f64");

        // Cross-op different op-kinds never collide.
        assert_ne!(key_b1, key_t1);
    }
}
