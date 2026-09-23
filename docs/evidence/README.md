# Evidence

Measurements kept as a record, referenced by the report's experimental-setup section.

## `thermal_throttling_mps.log`

Partial run of Benchmark A on the development machine (MacBook Air M4, 16 GB, fanless, MPS backend,
on battery with Low Power Mode enabled). Per-task wall-clock times:

| task | time |
|---|---|
| `aquatic_life` | 21m13s |
| `flora_and_insects` | 23m49s |
| `household_and_people` | **1h27m10s** |

The same fine-tuning (`mammals`, identical settings) completed in **4m13s** on the same machine
after it had been left idle to cool. Profiling ruled out the code as the cause: preprocessing
accounts for 2% of a training step, and the isolated forward/backward/step matches the synthetic
benchmark at 585 ms/batch.

The cause is thermal throttling of a passively cooled chassis under sustained GPU load, and it is
progressive — throughput degrades as the run proceeds. This is why all task vectors used in the
study are produced on a single datacenter GPU instead: not for speed, but so that every model in
the comparison is trained under identical conditions.
