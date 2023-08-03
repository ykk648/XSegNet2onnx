from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.comparator import Comparator, CompareFunc



def main():
    # The OnnxrtRunner requires an ONNX-RT session.
    # We can use the SessionFromOnnx lazy loader to construct one easily:
    build_onnxrt_session = SessionFromOnnx("xseg_out.onnx")
    build_onnxrt_session2 = SessionFromOnnx("xseg.onnx")
    
    runners = [
        OnnxrtRunner(build_onnxrt_session),
        OnnxrtRunner(build_onnxrt_session2),
    ]

    # `Comparator.run()` will run each runner separately using synthetic input data and
    #   return a `RunResults` instance. See `polygraphy/comparator/struct.py` for details.
    #
    # TIP: To use custom input data, you can set the `data_loader` parameter in `Comparator.run()``
    #   to a generator or iterable that yields `Dict[str, np.ndarray]`.
    run_results = Comparator.run(runners)

    # `Comparator.compare_accuracy()` checks that outputs match between runners.
    #
    # TIP: The `compare_func` parameter can be used to control how outputs are compared (see API reference for details).
    #   The default comparison function is created by `CompareFunc.simple()`, but we can construct it
    #   explicitly if we want to change the default parameters, such as tolerance.
    assert bool(Comparator.compare_accuracy(run_results, compare_func=CompareFunc.simple(atol=1e-8)))

    # We can use `RunResults.save()` method to save the inference results to a JSON file.
    # This can be useful if you want to generate and compare results separately.
    run_results.save("inference_results.json")


if __name__ == "__main__":
    main()