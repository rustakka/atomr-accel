//! End-to-end test for the IPluginCreatorV3One proxy. Builds only
//! when both `tensorrt-link` and `tensorrt-plugin` are on; registers
//! a stub Rust plugin and verifies `register_plugin` returns Ok.

#![cfg(all(
    feature = "tensorrt-link",
    feature = "tensorrt-plugin",
    feature = "cuda-runtime-tests"
))]

use std::sync::Arc;

use atomr_accel_tensorrt::plugin::{register_plugin, PluginV3};

struct DemoPlugin {
    name: String,
    version: String,
}

impl PluginV3 for DemoPlugin {
    fn name(&self) -> &str {
        &self.name
    }
    fn version(&self) -> &str {
        &self.version
    }
    fn clone_boxed(&self) -> Box<dyn PluginV3> {
        Box::new(DemoPlugin {
            name: self.name.clone(),
            version: self.version.clone(),
        })
    }
}

#[test]
#[ignore = "requires libnvinfer + libnvinfer_plugin"]
fn register_demo_plugin() {
    atomr_accel_tensorrt::init_logger();
    let plugin: Arc<dyn PluginV3> = Arc::new(DemoPlugin {
        name: "AtomrAccelDemoPlugin".into(),
        version: "1".into(),
    });
    register_plugin(plugin).expect("register_plugin");
}
