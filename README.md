# Performance tests for Fairlearn

This repository exclusively contains performance tests for the [Fairlearn](https://github.com/fairlearn/fairlearn) repository.

## Running tests locally

Setup:

```bash
# clone Fairlearn repo and Fairlearn-Performance repo
git clone https://github.com/fairlearn/fairlearn.git
git clone https://github.com/fairlearn/fairlearn-performance.git
# install Fairlearn-Performance dependencies
pip install -r fairlearn-performance/requirements.txt
# install latest Fairlearn to ensure compatibility
pip install fairlearn/.

# to show all tests
python -m pytest .\fairlearn-performance\perf\ --maxfail=1 -s --collect-only
# to run a single test from the full list
python -m pytest .\fairlearn-performance\perf\ --maxfail=1 -s -k test_perf[[dataset:adult_uci,estimator:DecisionTreeClassifier[],mitigator:GridSearch,disparity_metric:TruePositiveRateParity[]]]
```
