# Kunlunxin P800 35B gates at b560e33

Qwen3.6-35B-A3B BF16 TP4 passed 25/26 with eager and 25/26 in graph mode at the exact plugin commit b560e3309e821e08e38854aae6670a15402e39ce. Both modes returned only HELLO in the unchanged image-single OCR case and passed every concurrent image and mixed request. The graph run recorded capture and request-time FULL execution. Both serving logs had no NoC timeout.

The same host, container, checked-in dispatch policy, automatic Runner V2, and launch flags from the exact 27B results were used. Each directory contains the five original scenario reports and a runtime excerpt.

35B model metadata SHA256: config 93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99, weight index 41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83.
