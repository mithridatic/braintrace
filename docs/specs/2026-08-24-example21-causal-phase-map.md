# Example 21: causal explanation

## What the model is

A model is a set of mathematical operations and learned numbers. It receives
an input and calculates output scores. Training changes the learned numbers so
that correct outputs get higher scores.

The current direct model has two MiniLSTM memories. Each memory has 128 hidden
units. One memory reads the demonstration pairs. The other reads the test input
and the decode events. The model combines both memories into a 384-value task
state. Twelve color experts and 16 routing programs use this state. Separate
output heads calculate scores for the height, width, and colors of the answer.

The model is not one simple stack of layers. It has four stages: input-event
encoding, two parallel recurrent memory layers, relation and routing
calculations, and the three output heads.

This model does not contain spiking neurons. Its 256 MiniLSTM hidden units are
numerical state variables. They are sometimes called units or recurrent cells,
but they are not biological neurons.

Most of the model's 12.53 million trainable values are in dense connection
matrices. The model does not have a stored sparse graph of neuron-to-neuron
edges. Therefore it has no explicit synapse count.

No Dale sign constraint is applied to this model. Dale's law would require each
neuron to send connections with one sign: positive for an excitatory neuron or
negative for an inhibitory neuron. The current MiniLSTM weights can have either
sign. Older Example 21 spiking models support explicit edges and optional Dale
signs. The current direct model does not.

## What an ARC task asks the model to do

An ARC task supplies a few demonstration pairs. Each pair has an input grid and
its correct output grid. The task then supplies a new input grid without its
answer. The model must use the demonstrations to make the missing output grid.

An ARC grid is a rectangle of colored squares. Each square is a grid cell. A
cell contains one integer from 0 through 9. The integer identifies its color.
Color 0 is normally shown as black and often acts as the background.

The output shape is its number of rows and columns. Height is the row count.
Width is the column count. Shape matters because an ARC transformation can
crop, enlarge, rotate, or join grids. The output does not always have the same
shape as the input. A grid with the wrong height or width cannot be an exact
answer, even when all cells in the overlapping area have the correct colors.

The ARC answer uses exact integer values. Height can be any integer from 1
through 30, so it has 30 possible values. Width has the same 30 possible
values. Each cell can contain one of 10 colors, from 0 through 9. A grid can
therefore contain from 1 through 900 cells. A fixed 30 by 30 grid has
\(10^{900}\) possible color arrangements. Across every permitted shape, the
number of distinct grids is

\[
\sum_{h=1}^{30}\sum_{w=1}^{30}10^{hw}.
\]

The model uses `float32` values to calculate its scores. A `float32` value has
about seven significant decimal digits. These scores are approximate numeric
values, but the submitted prediction is discrete and exact: one integer
height, one integer width, and one integer color for every included cell.

The word *cell* has two meanings here. An ARC grid cell is one scored output
square. A MiniLSTM cell is an internal recurrent calculation. Only the ARC grid
cells appear in the prediction and are compared with the answer.

## What a prediction is

A prediction is the single output grid that the model submits for one test
input. It includes the chosen height, the chosen width, and one chosen color
for every cell inside that shape.

A strict task is correct only when the first prediction is exact for every test
input in that task.

The prediction is model-generated only when the trained model calculates all
of these choices. A forest, rule, template, retrieval system, repair step, or
reranker cannot make or select the grid.

## How the model calculates a prediction

1. The encoder converts the demonstrations and test input into a sequence of
   numeric events. The hidden test output is not part of this sequence.
2. The demonstration MiniLSTM updates while it reads demonstration events. The
   query MiniLSTM updates while it reads query and decode events.
3. The model joins the demonstration state, the query state, and their
   elementwise product. This forms the 384-value task state.
4. The color experts use the task state and the test-input colors to calculate
   ten scores for every possible output position. The routing programs add
   color votes from selected test-input positions. A score is also called a
   logit. It is a value used for comparison, not a finished probability.
5. The height head calculates 30 scores, one for each possible height from 1
   through 30. The width head does the same for width. The model produces these
   scores at all 30 decode rows. The decoder averages the scores and chooses
   the largest height score and the largest width score.
6. For each possible output cell, the decoder first checks the background
   gate. A non-positive gate selects color 0. A positive gate selects the
   largest of the nine scores for colors 1 through 9.
7. The decoder keeps only the rows and columns inside the selected height and
   width. The resulting rectangle is the first and only prediction.

## Does a perfect output representation score every task correctly?

Yes. A target-fed scorer oracle tested this directly on the fixed evaluation
set. The oracle used the hidden answers to construct uniquely correct height,
width, and color logits. It passed those logits through the production greedy
decoder and strict scorer.

The run loaded 400 tasks with 419 test queries. All 419 decoded grids matched
their targets. The scorer reported 400 correct tasks and zero failed tasks.
Oracle computation took 6.261 seconds. The complete Docker command took 18.547
seconds, including container startup.

This result proves that correct height, width, and color logits are sufficient
for the decoder and scorer. It is not model performance. The oracle read the
answers, while the model is not allowed to read them.

## What the trained model does now

The trained model can generate some exact synthetic answers. It can copy grids
and solve some small counting, labeling, and selection tasks. Its routing path
has generated one exact synthetic upscale answer.

The same model solved zero strict tasks in both recorded real ARC development
scopes. The complete evaluation set was not run with this model. It therefore
has no qualifying model-generated ARC result.

The failed predictions show two main problems. For spatial tasks, the model
does not reliably move the correct test-input content to each output position.
For some small label tasks, it predicts the correct one-cell shape but chooses
the wrong color. For crop and other shape-changing tasks, it can also choose
the wrong height or width.

These observations place the main failure between the internal task state and
the exact output grid. The state contains some useful task information, but the
height, width, and color heads do not turn that information into exact real ARC
predictions often enough. Some tasks can also fail earlier if the recurrent
memories do not identify the demonstrated relation.

Measurements, artifact paths, qualification rules, and the scorer-oracle
result are in
[`2026-08-24-example21-causal-evidence.md`](2026-08-24-example21-causal-evidence.md).
