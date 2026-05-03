//! `CudnnActor` — wraps a [`cudarc::cudnn::Cudnn`] handle and exposes
//! the most common neural-net operations as messages.
//!
//! cuDNN's `Cudnn` handle is `!Send + !Sync` per NVIDIA's threading
//! docs. This actor is constructed and run by `ContextActor` whose
//! cell sits on the [`crate::dispatcher::GpuDispatcher`] single
//! pinned thread, so we wrap the handle in a Send/Sync newtype with
//! `unsafe impl`s — same pattern as `RngActor`.
//!
//! F2 scope: `ConvForward` (NCHW f32), `ActivationForward` (RELU,
//! SIGMOID, TANH), `SoftmaxForward` (instance mode). cuDNN has many
//! more ops; we add them as patterns demand.
//!
//! Descriptor / algorithm / workspace caches:
//! - Tensor + filter + conv descriptors are LRU-cached keyed on
//!   shape + dtype + layout (256 entries each).
//! - Conv-forward algorithm is selected once per (input, filter,
//!   output, conv) shape and cached.
//! - Workspace `CudaSlice<u8>` grows on demand; never shrunk;
//!   rebuilt fresh on context restart.

use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::cudnn::{
    sys as cudnn_sys, ActivationDescriptor, ActivationForward, ConvDescriptor, ConvForward, Cudnn,
    FilterDescriptor, Softmax, SoftmaxForward, TensorDescriptor,
};
use cudarc::driver::CudaSlice;
use lru::LruCache;
use parking_lot::Mutex;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::stream::StreamAllocator;

const LIB: &str = "cudnn";
const DESCRIPTOR_CACHE_SIZE: usize = 256;

/// Convolution parameters (cuDNN 2D conv subset).
#[derive(Debug, Clone, Copy)]
pub struct ConvParams {
    pub pad: [i32; 2],
    pub stride: [i32; 2],
    pub dilation: [i32; 2],
}

/// Supported activation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationKind {
    Relu,
    Sigmoid,
    Tanh,
}

impl ActivationKind {
    fn cudnn_mode(self) -> cudnn_sys::cudnnActivationMode_t {
        match self {
            ActivationKind::Relu => cudnn_sys::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            ActivationKind::Sigmoid => cudnn_sys::cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID,
            ActivationKind::Tanh => cudnn_sys::cudnnActivationMode_t::CUDNN_ACTIVATION_TANH,
        }
    }
}

pub struct ConvForwardRequest {
    pub x: GpuRef<f32>,
    pub x_dims: [i32; 4], // NCHW
    pub w: GpuRef<f32>,
    pub w_dims: [i32; 4], // KCRS
    pub y: GpuRef<f32>,
    pub y_dims: [i32; 4], // NKHW
    pub conv: ConvParams,
    pub alpha: f32,
    pub beta: f32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

pub struct ActivationRequest {
    pub kind: ActivationKind,
    pub x: GpuRef<f32>,
    pub y: GpuRef<f32>,
    pub dims: [i32; 4],
    pub alpha: f32,
    pub beta: f32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

pub struct SoftmaxRequest {
    pub x: GpuRef<f32>,
    pub y: GpuRef<f32>,
    pub dims: [i32; 4],
    pub alpha: f32,
    pub beta: f32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

pub enum CudnnMsg {
    ConvForward(Box<ConvForwardRequest>),
    Activation(Box<ActivationRequest>),
    Softmax(Box<SoftmaxRequest>),
}

pub struct CudnnActor {
    inner: CudnnInner,
}

struct SendCudnn(Arc<Cudnn>);
unsafe impl Send for SendCudnn {}
unsafe impl Sync for SendCudnn {}

/// Generic Send/Sync newtype around an `Arc<T>` where `T` carries
/// cuDNN handles that are `!Send + !Sync` only because they hold raw
/// FFI pointers. The actor running them is pinned to one OS thread
/// via [`crate::dispatcher::GpuDispatcher`].
#[repr(transparent)]
struct SendDesc<T>(Arc<T>);
unsafe impl<T> Send for SendDesc<T> {}
unsafe impl<T> Sync for SendDesc<T> {}
impl<T> Clone for SendDesc<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> std::ops::Deref for SendDesc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct TensorKey {
    dims: [i32; 4],
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct FilterKey {
    dims: [i32; 4],
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct ConvKey {
    pad: [i32; 2],
    stride: [i32; 2],
    dilation: [i32; 2],
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct ConvAlgoKey {
    x: TensorKey,
    w: FilterKey,
    y: TensorKey,
    conv: ConvKey,
}

struct DescriptorCache {
    tensors: LruCache<TensorKey, SendDesc<TensorDescriptor<f32>>>,
    filters: LruCache<FilterKey, SendDesc<FilterDescriptor<f32>>>,
    convs: LruCache<ConvKey, SendDesc<ConvDescriptor<f32>>>,
    activations: LruCache<ActivationKind, SendDesc<ActivationDescriptor<f32>>>,
    softmax: Option<SendDesc<Softmax<f32>>>,
    algos: LruCache<ConvAlgoKey, cudnn_sys::cudnnConvolutionFwdAlgo_t>,
}

impl DescriptorCache {
    fn new() -> Self {
        let cap = NonZeroUsize::new(DESCRIPTOR_CACHE_SIZE).unwrap();
        Self {
            tensors: LruCache::new(cap),
            filters: LruCache::new(cap),
            convs: LruCache::new(cap),
            activations: LruCache::new(NonZeroUsize::new(8).unwrap()),
            softmax: None,
            algos: LruCache::new(cap),
        }
    }
}

unsafe impl Send for DescriptorCache {}
unsafe impl Sync for DescriptorCache {}

enum CudnnInner {
    Real {
        handle: SendCudnn,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        descriptors: Mutex<DescriptorCache>,
        workspace: Mutex<Option<CudaSlice<u8>>>,
        #[allow(dead_code)]
        state: Arc<DeviceState>,
    },
    Mock,
}

impl CudnnActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
    ) -> Props<Self> {
        Props::create(move || {
            let handle = match Cudnn::new(stream.clone()) {
                Ok(h) => h,
                Err(e) => panic!("ContextPoisoned: Cudnn::new failed: {e}"),
            };
            CudnnActor {
                inner: CudnnInner::Real {
                    handle: SendCudnn(handle),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    descriptors: Mutex::new(DescriptorCache::new()),
                    workspace: Mutex::new(None),
                    state: state.clone(),
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| CudnnActor {
            inner: CudnnInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for CudnnActor {
    type Msg = CudnnMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: CudnnMsg) {
        match &self.inner {
            CudnnInner::Mock => reply_mock(msg),
            CudnnInner::Real {
                handle,
                stream,
                completion,
                descriptors,
                workspace,
                ..
            } => match msg {
                CudnnMsg::ConvForward(req) => {
                    handle_conv_forward(
                        &handle.0,
                        stream,
                        completion,
                        descriptors,
                        workspace,
                        *req,
                    );
                }
                CudnnMsg::Activation(req) => {
                    handle_activation(&handle.0, stream, completion, descriptors, *req);
                }
                CudnnMsg::Softmax(req) => {
                    handle_softmax(&handle.0, stream, completion, descriptors, *req);
                }
            },
        }
    }
}

fn reply_mock(msg: CudnnMsg) {
    let err = || GpuError::Unrecoverable("CudnnActor in mock mode".into());
    match msg {
        CudnnMsg::ConvForward(r) => {
            let _ = r.reply.send(Err(err()));
        }
        CudnnMsg::Activation(r) => {
            let _ = r.reply.send(Err(err()));
        }
        CudnnMsg::Softmax(r) => {
            let _ = r.reply.send(Err(err()));
        }
    }
}

fn get_or_make_tensor(
    handle: &Arc<Cudnn>,
    cache: &mut DescriptorCache,
    key: TensorKey,
) -> Result<SendDesc<TensorDescriptor<f32>>, GpuError> {
    if let Some(t) = cache.tensors.get(&key) {
        return Ok(t.clone());
    }
    let t = handle
        .create_4d_tensor::<f32>(cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, key.dims)
        .map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("create_4d_tensor: {e}"),
        })?;
    let t = SendDesc(Arc::new(t));
    cache.tensors.put(key, t.clone());
    Ok(t)
}

fn get_or_make_filter(
    handle: &Arc<Cudnn>,
    cache: &mut DescriptorCache,
    key: FilterKey,
) -> Result<SendDesc<FilterDescriptor<f32>>, GpuError> {
    if let Some(f) = cache.filters.get(&key) {
        return Ok(f.clone());
    }
    let f = handle
        .create_4d_filter::<f32>(cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, key.dims)
        .map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("create_4d_filter: {e}"),
        })?;
    let f = SendDesc(Arc::new(f));
    cache.filters.put(key, f.clone());
    Ok(f)
}

fn get_or_make_conv(
    handle: &Arc<Cudnn>,
    cache: &mut DescriptorCache,
    key: ConvKey,
) -> Result<SendDesc<ConvDescriptor<f32>>, GpuError> {
    if let Some(c) = cache.convs.get(&key) {
        return Ok(c.clone());
    }
    let c = handle
        .create_conv2d::<f32>(
            key.pad,
            key.stride,
            key.dilation,
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )
        .map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("create_conv2d: {e}"),
        })?;
    let c = SendDesc(Arc::new(c));
    cache.convs.put(key, c.clone());
    Ok(c)
}

fn get_or_make_activation(
    handle: &Arc<Cudnn>,
    cache: &mut DescriptorCache,
    kind: ActivationKind,
) -> Result<SendDesc<ActivationDescriptor<f32>>, GpuError> {
    if let Some(a) = cache.activations.get(&kind) {
        return Ok(a.clone());
    }
    let a = handle
        .create_activation::<f32>(
            kind.cudnn_mode(),
            cudnn_sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            f64::MAX,
        )
        .map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("create_activation: {e}"),
        })?;
    let a = SendDesc(Arc::new(a));
    cache.activations.put(kind, a.clone());
    Ok(a)
}

fn handle_conv_forward(
    handle: &Arc<Cudnn>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    descriptors: &Mutex<DescriptorCache>,
    workspace: &Mutex<Option<CudaSlice<u8>>>,
    req: ConvForwardRequest,
) {
    let ConvForwardRequest {
        x,
        x_dims,
        w,
        w_dims,
        y,
        y_dims,
        conv: cp,
        alpha,
        beta,
        reply,
    } = req;

    let (x_slice, w_slice, y_slice) = match envelope::access_all_3(&x, &w, &y) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut y_owned = match Arc::try_unwrap(y_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Conv output Y has multiple live references".into(),
            )));
            return;
        }
    };

    let x_key = TensorKey { dims: x_dims };
    let w_key = FilterKey { dims: w_dims };
    let y_key = TensorKey { dims: y_dims };
    let c_key = ConvKey {
        pad: cp.pad,
        stride: cp.stride,
        dilation: cp.dilation,
    };
    let algo_key = ConvAlgoKey {
        x: x_key,
        w: w_key,
        y: y_key,
        conv: c_key,
    };

    let (x_desc, w_desc, y_desc, c_desc, algo, ws_size) = {
        let mut cache = descriptors.lock();
        let x_desc = match get_or_make_tensor(handle, &mut cache, x_key) {
            Ok(d) => d,
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        let w_desc = match get_or_make_filter(handle, &mut cache, w_key) {
            Ok(d) => d,
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        let y_desc = match get_or_make_tensor(handle, &mut cache, y_key) {
            Ok(d) => d,
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        let c_desc = match get_or_make_conv(handle, &mut cache, c_key) {
            Ok(d) => d,
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };

        // Pick algo (cached).
        let algo = if let Some(a) = cache.algos.get(&algo_key) {
            *a
        } else {
            let op = ConvForward {
                conv: &*c_desc,
                x: &*x_desc,
                w: &*w_desc,
                y: &*y_desc,
            };
            match op.pick_algorithm() {
                Ok(a) => {
                    cache.algos.put(algo_key, a);
                    a
                }
                Err(e) => {
                    let _ = reply.send(Err(GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("pick_algorithm: {e}"),
                    }));
                    return;
                }
            }
        };

        // Workspace size for this (op, algo).
        let op = ConvForward {
            conv: &*c_desc,
            x: &*x_desc,
            w: &*w_desc,
            y: &*y_desc,
        };
        let ws_size = match op.get_workspace_size(algo) {
            Ok(s) => s,
            Err(e) => {
                let _ = reply.send(Err(GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("get_workspace_size: {e}"),
                }));
                return;
            }
        };
        (x_desc, w_desc, y_desc, c_desc, algo, ws_size)
    };

    // Grow workspace if needed.
    {
        let mut ws_lock = workspace.lock();
        let need_alloc = match ws_lock.as_ref() {
            None => ws_size > 0,
            Some(slice) => slice.num_bytes() < ws_size,
        };
        if need_alloc {
            match stream.alloc_zeros::<u8>(ws_size) {
                Ok(s) => {
                    *ws_lock = Some(s);
                }
                Err(e) => {
                    let _ = reply.send(Err(GpuError::OutOfMemory(format!(
                        "cudnn workspace ({ws_size}B): {e}"
                    ))));
                    return;
                }
            }
        }
    }

    y.record_write(stream);

    // Hold descriptor Arcs in keep_alive so they outlive the kernel.
    let descriptors_arc = (
        x_desc.clone(),
        w_desc.clone(),
        y_desc.clone(),
        c_desc.clone(),
    );
    let workspace_clone = workspace; // borrow for closure
    let stream_for_clos = stream.clone();
    let workspace_ptr: *const Mutex<Option<CudaSlice<u8>>> = workspace_clone;

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        // SAFETY: `workspace_ptr` is valid for the lifetime of the
        // ContextActor that owns it; the closure runs synchronously
        // before `run_kernel` returns. We acquire the mutex briefly
        // to take a temporary &mut into the workspace.
        let workspace_mutex: &Mutex<Option<CudaSlice<u8>>> = unsafe { &*workspace_ptr };
        let mut ws_lock = workspace_mutex.lock();
        let op = ConvForward {
            conv: &*descriptors_arc.3,
            x: &*descriptors_arc.0,
            w: &*descriptors_arc.1,
            y: &*descriptors_arc.2,
        };
        let res = unsafe {
            op.launch::<CudaSlice<u8>, _, _, _>(
                algo,
                ws_lock.as_mut(),
                (alpha, beta),
                &*x_slice,
                &*w_slice,
                &mut y_owned,
            )
        };
        drop(ws_lock);
        let _ = stream_for_clos; // silence move check
        res.map(|_| (x_slice, w_slice, y_owned, descriptors_arc))
            .map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("conv_forward launch: {e}"),
            })
    });
}

fn handle_activation(
    handle: &Arc<Cudnn>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    descriptors: &Mutex<DescriptorCache>,
    req: ActivationRequest,
) {
    let ActivationRequest {
        kind,
        x,
        y,
        dims,
        alpha,
        beta,
        reply,
    } = req;
    let (x_slice, y_slice) = match envelope::access_all_2(&x, &y) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut y_owned = match Arc::try_unwrap(y_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Activation y has multiple live references".into(),
            )));
            return;
        }
    };
    let key = TensorKey { dims };
    let (x_desc, y_desc, act_desc) = {
        let mut cache = descriptors.lock();
        let x_desc = match get_or_make_tensor(handle, &mut cache, key) {
            Ok(d) => d,
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        let y_desc = x_desc.clone();
        let act_desc = match get_or_make_activation(handle, &mut cache, kind) {
            Ok(d) => d,
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        (x_desc, y_desc, act_desc)
    };
    y.record_write(stream);
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let op = ActivationForward {
            act: &*act_desc,
            x: &*x_desc,
            y: &*y_desc,
        };
        unsafe { op.launch((alpha, beta), &*x_slice, &mut y_owned) }
            .map(|_| (x_slice, y_owned, act_desc, x_desc, y_desc))
            .map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("activation launch: {e}"),
            })
    });
}

fn handle_softmax(
    handle: &Arc<Cudnn>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    descriptors: &Mutex<DescriptorCache>,
    req: SoftmaxRequest,
) {
    let SoftmaxRequest {
        x,
        y,
        dims,
        alpha,
        beta,
        reply,
    } = req;
    let (x_slice, y_slice) = match envelope::access_all_2(&x, &y) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut y_owned = match Arc::try_unwrap(y_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Softmax y has multiple live references".into(),
            )));
            return;
        }
    };
    let key = TensorKey { dims };
    let (x_desc, y_desc, sm_desc) = {
        let mut cache = descriptors.lock();
        let x_desc = match get_or_make_tensor(handle, &mut cache, key) {
            Ok(d) => d,
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        let y_desc = x_desc.clone();
        let sm_desc = if let Some(s) = cache.softmax.clone() {
            s
        } else {
            match handle
                .create_softmax::<f32>(cudnn_sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE)
            {
                Ok(s) => {
                    let s = SendDesc(Arc::new(s));
                    cache.softmax = Some(s.clone());
                    s
                }
                Err(e) => {
                    let _ = reply.send(Err(GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("create_softmax: {e}"),
                    }));
                    return;
                }
            }
        };
        (x_desc, y_desc, sm_desc)
    };
    y.record_write(stream);
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let op = SoftmaxForward {
            softmax: &*sm_desc,
            x: &*x_desc,
            y: &*y_desc,
        };
        unsafe {
            op.launch(
                (alpha, beta),
                cudnn_sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
                &*x_slice,
                &mut y_owned,
            )
        }
        .map(|_| (x_slice, y_owned, sm_desc, x_desc, y_desc))
        .map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("softmax launch: {e}"),
        })
    });
}
