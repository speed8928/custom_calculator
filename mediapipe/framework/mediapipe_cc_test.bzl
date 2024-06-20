"""Macro for multi-platform C++ tests."""

# buildifier: disable=out-of-order-load

DEFAULT_ADDITIONAL_TEST_DEPS = []

def mediapipe_cc_test(
        name,
        srcs = [],
        data = [],
        deps = [],
        size = None,
        tags = [],
        timeout = None,
        args = [],
        additional_deps = DEFAULT_ADDITIONAL_TEST_DEPS,
        platforms = ["linux", "android", "ios", "wasm"],
        exclude_platforms = None,
        # ios_unit_test arguments
        ios_minimum_os_version = "12.0",
        # android_cc_test arguments
        open_gl_driver = None,
        emulator_mini_boot = True,
        requires_full_emulation = True,
        android_devices = {},
        # wasm_web_test arguments
        browsers = None,
        **kwargs):
    native.cc_library(
        name = name + "_lib",
        testonly = 1,
        srcs = srcs,
        data = data,
        deps = deps + additional_deps,
        alwayslink = 1,
    )

    native.cc_test(
        name = name,
        size = size,
        timeout = timeout,
        deps = [":{}_lib".format(name)],
    )
