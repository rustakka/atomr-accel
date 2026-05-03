//! `TensorActor` — wraps cuTENSOR for Einstein-summation contractions.
//!
//! cudarc 0.19 exposes cuTENSOR at sys + a thin `result.rs` wrapper.
//! The compute-descriptor argument required by `cutensorCreate*` is a
//! predefined extern global in libcutensor.so (`CUTENSOR_R_MIN_32F`
//! etc.). We link against the symbol directly via `extern "C"`.
//!
//! Supported ops (F-phase 9.x, sys-level):
//! - `Contract { a, b, c, modes_a, modes_b, modes_c, alpha, beta }` —
//!   D = alpha * A^modes_a · B^modes_b + beta * C^modes_c
//!   (D is written in-place into the C buffer).

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::cutensor::result as ct_result;
use cudarc::cutensor::sys as ct_sys;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use parking_lot::Mutex;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::stream::StreamAllocator;

const LIB: &str = "cutensor";

// Predefined compute descriptor for f32 with min-precision arithmetic.
// libcutensor exports this as an extern global; we declare it here.
extern "C" {
    static CUTENSOR_R_MIN_32F: ct_sys::cutensorComputeDescriptor_t;
}

fn r_min_32f() -> ct_sys::cutensorComputeDescriptor_t {
    unsafe { CUTENSOR_R_MIN_32F }
}

/// Specification of one tensor in a contraction call.
#[derive(Clone)]
pub struct TensorSpec {
    pub buf: GpuRef<f32>,
    /// Per-mode extents (length = num_modes).
    pub extent: Vec<i64>,
    /// Per-mode strides (length = num_modes). If empty, treated as
    /// dense column-major.
    pub stride: Vec<i64>,
    /// Per-mode labels (Einstein summation indices).
    pub modes: Vec<i32>,
}

pub enum TensorMsg {
    /// D = alpha * A · B + beta * C.  D is written in-place into `c`.
    /// Mode labels follow Einstein notation: any mode appearing in
    /// both `a.modes` and `b.modes` but not `c.modes` is contracted.
    Contract {
        a: TensorSpec,
        b: TensorSpec,
        c: TensorSpec,
        alpha: f32,
        beta: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct TensorActor {
    inner: TensorInner,
}

struct SendHandle(ct_sys::cutensorHandle_t);
unsafe impl Send for SendHandle {}
unsafe impl Sync for SendHandle {}

#[allow(dead_code)]
enum TensorInner {
    Real {
        handle: Mutex<SendHandle>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        workspace: Mutex<Option<CudaSlice<u8>>>,
    },
    Mock,
}

impl Drop for TensorInner {
    fn drop(&mut self) {
        if let TensorInner::Real { handle, .. } = self {
            let h = handle.lock();
            unsafe {
                let _ = ct_result::destroy_handle(h.0);
            }
        }
    }
}

impl TensorActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
    ) -> Props<Self> {
        Props::create(move || {
            let h = match ct_result::create_handle() {
                Ok(h) => h,
                Err(e) => panic!("ContextPoisoned: cutensorCreate failed: {e}"),
            };
            TensorActor {
                inner: TensorInner::Real {
                    handle: Mutex::new(SendHandle(h)),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
                    workspace: Mutex::new(None),
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| TensorActor {
            inner: TensorInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for TensorActor {
    type Msg = TensorMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: TensorMsg) {
        match &self.inner {
            TensorInner::Mock => mock_reply(msg),
            TensorInner::Real {
                handle,
                stream,
                completion,
                workspace,
                ..
            } => match msg {
                TensorMsg::Contract {
                    a,
                    b,
                    c,
                    alpha,
                    beta,
                    reply,
                } => {
                    handle_contract(
                        handle, stream, completion, workspace, a, b, c, alpha, beta, reply,
                    );
                }
            },
        }
    }
}

fn mock_reply(msg: TensorMsg) {
    let err = || GpuError::Unrecoverable("TensorActor in mock mode".into());
    match msg {
        TensorMsg::Contract { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
    }
}

fn ensure_workspace(
    workspace: &Mutex<Option<CudaSlice<u8>>>,
    stream: &Arc<cudarc::driver::CudaStream>,
    needed_bytes: usize,
) -> Result<(), GpuError> {
    let mut g = workspace.lock();
    let cur = g.as_ref().map(|s| s.len()).unwrap_or(0);
    if cur >= needed_bytes {
        return Ok(());
    }
    *g = Some(stream.alloc_zeros::<u8>(needed_bytes.max(1)).map_err(|e| {
        GpuError::OutOfMemory(format!("cutensor workspace ({needed_bytes}B): {e}"))
    })?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_contract(
    handle: &Mutex<SendHandle>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    workspace: &Mutex<Option<CudaSlice<u8>>>,
    a: TensorSpec,
    b: TensorSpec,
    c: TensorSpec,
    alpha: f32,
    beta: f32,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
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

    let h = handle.lock();
    // Pointers are re-fetched inside the run_kernel closure; here we
    // only need the lifetime guards to outlive descriptor creation.
    let (_a_ptr, _ga) = a_slice.device_ptr(stream);
    let (_b_ptr, _gb) = b_slice.device_ptr(stream);
    let (_c_ptr, _gc) = c_owned.device_ptr_mut(stream);

    // Build tensor descriptors.
    let a_stride = if a.stride.is_empty() {
        std::ptr::null()
    } else {
        a.stride.as_ptr()
    };
    let b_stride = if b.stride.is_empty() {
        std::ptr::null()
    } else {
        b.stride.as_ptr()
    };
    let c_stride = if c.stride.is_empty() {
        std::ptr::null()
    } else {
        c.stride.as_ptr()
    };

    let desc_a = match unsafe {
        ct_result::create_tensor_descriptor(
            h.0,
            a.extent.len() as u32,
            a.extent.as_ptr(),
            a_stride,
            ct_sys::cudaDataType_t::CUDA_R_32F,
            16,
        )
    } {
        Ok(d) => d,
        Err(e) => {
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("CreateTensorDescriptor(A): {e}"),
            }));
            return;
        }
    };
    let desc_b = match unsafe {
        ct_result::create_tensor_descriptor(
            h.0,
            b.extent.len() as u32,
            b.extent.as_ptr(),
            b_stride,
            ct_sys::cudaDataType_t::CUDA_R_32F,
            16,
        )
    } {
        Ok(d) => d,
        Err(e) => {
            unsafe {
                let _ = ct_result::destroy_tensor_descriptor(desc_a);
            }
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("CreateTensorDescriptor(B): {e}"),
            }));
            return;
        }
    };
    let desc_c = match unsafe {
        ct_result::create_tensor_descriptor(
            h.0,
            c.extent.len() as u32,
            c.extent.as_ptr(),
            c_stride,
            ct_sys::cudaDataType_t::CUDA_R_32F,
            16,
        )
    } {
        Ok(d) => d,
        Err(e) => {
            unsafe {
                let _ = ct_result::destroy_tensor_descriptor(desc_b);
                let _ = ct_result::destroy_tensor_descriptor(desc_a);
            }
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("CreateTensorDescriptor(C): {e}"),
            }));
            return;
        }
    };

    let op_desc = match unsafe {
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
            r_min_32f(),
        )
    } {
        Ok(d) => d,
        Err(e) => {
            unsafe {
                let _ = ct_result::destroy_tensor_descriptor(desc_c);
                let _ = ct_result::destroy_tensor_descriptor(desc_b);
                let _ = ct_result::destroy_tensor_descriptor(desc_a);
            }
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("CreateContraction: {e}"),
            }));
            return;
        }
    };

    let pref = match unsafe {
        ct_result::create_plan_preference(
            h.0,
            ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
            ct_sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
        )
    } {
        Ok(p) => p,
        Err(e) => {
            unsafe {
                let _ = ct_result::destroy_operation_descriptor(op_desc);
                let _ = ct_result::destroy_tensor_descriptor(desc_c);
                let _ = ct_result::destroy_tensor_descriptor(desc_b);
                let _ = ct_result::destroy_tensor_descriptor(desc_a);
            }
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("CreatePlanPreference: {e}"),
            }));
            return;
        }
    };

    let ws_size = match unsafe {
        ct_result::estimate_workspace_size(
            h.0,
            op_desc,
            pref,
            ct_sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_DEFAULT,
        )
    } {
        Ok(s) => s,
        Err(e) => {
            unsafe {
                let _ = ct_result::destroy_plan_preference(pref);
                let _ = ct_result::destroy_operation_descriptor(op_desc);
                let _ = ct_result::destroy_tensor_descriptor(desc_c);
                let _ = ct_result::destroy_tensor_descriptor(desc_b);
                let _ = ct_result::destroy_tensor_descriptor(desc_a);
            }
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("EstimateWorkspaceSize: {e}"),
            }));
            return;
        }
    };

    let plan = match unsafe { ct_result::create_plan(h.0, op_desc, pref, ws_size) } {
        Ok(p) => p,
        Err(e) => {
            unsafe {
                let _ = ct_result::destroy_plan_preference(pref);
                let _ = ct_result::destroy_operation_descriptor(op_desc);
                let _ = ct_result::destroy_tensor_descriptor(desc_c);
                let _ = ct_result::destroy_tensor_descriptor(desc_b);
                let _ = ct_result::destroy_tensor_descriptor(desc_a);
            }
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("CreatePlan: {e}"),
            }));
            return;
        }
    };

    drop((_ga, _gb, _gc));
    drop(h);

    if let Err(e) = ensure_workspace(workspace, stream, ws_size as usize) {
        unsafe {
            let _ = ct_result::destroy_plan(plan);
            let _ = ct_result::destroy_plan_preference(pref);
            let _ = ct_result::destroy_operation_descriptor(op_desc);
            let _ = ct_result::destroy_tensor_descriptor(desc_c);
            let _ = ct_result::destroy_tensor_descriptor(desc_b);
            let _ = ct_result::destroy_tensor_descriptor(desc_a);
        }
        let _ = reply.send(Err(e));
        return;
    }

    c.buf.record_write(stream);

    let handle_clone = handle;
    let workspace_ref = workspace;
    let stream_for_check = stream.clone();

    struct OpGuard {
        plan: ct_sys::cutensorPlan_t,
        pref: ct_sys::cutensorPlanPreference_t,
        op: ct_sys::cutensorOperationDescriptor_t,
        a: ct_sys::cutensorTensorDescriptor_t,
        b: ct_sys::cutensorTensorDescriptor_t,
        c: ct_sys::cutensorTensorDescriptor_t,
    }
    impl Drop for OpGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = ct_result::destroy_plan(self.plan);
                let _ = ct_result::destroy_plan_preference(self.pref);
                let _ = ct_result::destroy_operation_descriptor(self.op);
                let _ = ct_result::destroy_tensor_descriptor(self.c);
                let _ = ct_result::destroy_tensor_descriptor(self.b);
                let _ = ct_result::destroy_tensor_descriptor(self.a);
            }
        }
    }
    unsafe impl Send for OpGuard {}
    let guard = OpGuard {
        plan,
        pref,
        op: op_desc,
        a: desc_a,
        b: desc_b,
        c: desc_c,
    };

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle_clone.lock();
        let mut ws = workspace_ref.lock();
        let (a_ptr, _ga) = a_slice.device_ptr(&stream_for_check);
        let (b_ptr, _gb) = b_slice.device_ptr(&stream_for_check);
        let (c_ptr, _gc) = c_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _gws) = ws_slice.device_ptr_mut(&stream_for_check);
        let alpha_h = alpha;
        let beta_h = beta;
        let res = unsafe {
            ct_result::contract(
                h.0,
                guard.plan,
                &alpha_h as *const f32 as *const _,
                a_ptr as *const _,
                b_ptr as *const _,
                &beta_h as *const f32 as *const _,
                c_ptr as *const _,
                c_ptr as *mut _,
                ws_ptr as *mut _,
                ws_size,
                stream_for_check.cu_stream() as *mut _,
            )
        };
        drop((_ga, _gb, _gc, _gws));
        match res {
            Ok(()) => Ok((c_owned, a_slice, b_slice, guard)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Contract: {e}"),
            }),
        }
    });
}
