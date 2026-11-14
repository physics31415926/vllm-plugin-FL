# Kunlunxin P800 graph-shape comparison

All three series use plugin revision b560e33 with the same P800 TP4 host, model, image and checked-in dispatch policy. The earlier repeat with default sizes 1,2,4,8 passed 23 of 26, reproducing the same three concurrent image outputs stuck on punctuation. The subsequent matched two runs passed 26/26 and 33/33 additional batch-order requests with exact size 3 added and with default sizes.

Replay logs record three active rows mapped to 3 or 4 slots. Default padding is present in both passing and failing runs. The evidence associates the failure with prolonged runs of a padded shape but does not locate its cause.
