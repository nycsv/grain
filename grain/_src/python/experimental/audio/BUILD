# Experimental duration-based bucketing sampler for variable-length
# sequence workloads (speech/ASR, TTS, and similar).

package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "duration_bucketing_sampler",
    srcs = ["duration_bucketing_sampler.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:sharding",
        "//grain/_src/python:record",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "duration_bucketing_sampler_test",
    srcs = ["duration_bucketing_sampler_test.py"],
    srcs_version = "PY3",
    deps = [
        ":duration_bucketing_sampler",
        "//grain/_src/core:sharding",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)
