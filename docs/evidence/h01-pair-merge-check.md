# Merge check: pair 4157825456 / 5654281423

**Verdict: suspected merge: undetermined** (71 candidate contacts).

| Evidence | Value |
| --- | --- |
| Shared C3 ids by source | {"edge_list": [], "checkpoint:h01-resolved-edge-list-3000.cells.json": []} |
| Cells present in each source | {"edge_list": ["4157825456", "5654281423"], "checkpoint:h01-resolved-edge-list-3000.cells.json": []} |
| Direction and type mix | {"4157825456->type1": 5, "4157825456->type2": 9, "5654281423->type1": 14, "5654281423->type2": 43} |
| Endpoint reads (pre/post) | {"background/background": 70, "cell_a/background": 1} |
| Rows with one nonzero label at both ends | 0 |
| Rows with expected cells at both ends | 0 |
| Rows background at both ends | 70 |
| Rows within the 2-voxel box | 0 |

The saved JSON can show a merge (shared C3 label or one label at both ends) or an ordinary contact (expected cells at both ends); endpoints on background decide neither. Decisive follow-up: read the C3 label at each endpoint voxel and test membership in each cell's sampled c3_ids (online, one scattered_points call).

| Annotation | Pre | Post | Type | Pre reads | Post reads | Pre offset | Post offset |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100144637 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 102305670 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 103822813 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 10478123 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 108301250 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 108471882 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 109360923 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 112467397 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 114952934 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 116483373 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 11673508 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 117174015 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 118020119 | 4157825456 | 5654281423 | 1 | 0 (background) | 0 (background) | None | None |
| 120953297 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 121188567 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 121199763 | 4157825456 | 5654281423 | 1 | 0 (background) | 0 (background) | None | None |
| 12477026 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 125917741 | 4157825456 | 5654281423 | 2 | 0 (background) | 0 (background) | None | None |
| 126091183 | 4157825456 | 5654281423 | 2 | 0 (background) | 0 (background) | None | None |
| 126593420 | 4157825456 | 5654281423 | 2 | 0 (background) | 0 (background) | None | None |
| 127829892 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 129934473 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 130313441 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 136569133 | 4157825456 | 5654281423 | 1 | 4157825456 (cell_a) | 0 (background) | [0, 0, 0] | None |
| 141682179 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 143441773 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 14477816 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 145666376 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 146026453 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 146771403 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 16232487 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 16284236 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 17585238 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 19690278 | 4157825456 | 5654281423 | 2 | 0 (background) | 0 (background) | None | None |
| 20470362 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 2067517 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 34428532 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 34902484 | 4157825456 | 5654281423 | 1 | 0 (background) | 0 (background) | None | None |
| 43561472 | 4157825456 | 5654281423 | 2 | 0 (background) | 0 (background) | None | None |
| 47199952 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 47964556 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 53267084 | 4157825456 | 5654281423 | 2 | 0 (background) | 0 (background) | None | None |
| 59528985 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 59723160 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 59727895 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 62374479 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 63967920 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 63977941 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 66270803 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 68575152 | 4157825456 | 5654281423 | 2 | 0 (background) | 0 (background) | None | None |
| 68870053 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 71527686 | 4157825456 | 5654281423 | 2 | 0 (background) | 0 (background) | None | None |
| 71808175 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 72534918 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 73723691 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 75320335 | 4157825456 | 5654281423 | 2 | 0 (background) | 0 (background) | None | None |
| 77120253 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 81131902 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 82122302 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 84290094 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 85204895 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 85787101 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 87007535 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 87527105 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 89875505 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 91718495 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 92529008 | 4157825456 | 5654281423 | 1 | 0 (background) | 0 (background) | None | None |
| 93316465 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
| 94974176 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 96502623 | 5654281423 | 4157825456 | 2 | 0 (background) | 0 (background) | None | None |
| 99500285 | 5654281423 | 4157825456 | 1 | 0 (background) | 0 (background) | None | None |
