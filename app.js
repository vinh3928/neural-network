
//initalization of varibles

function main () {
var weights = [],
  values = [],
  expectedValues = [],
  thresholds = [];

  initalize(weights, values, expectedValues, thresholds)
  connectNodes(weights, thresholds)
  var counter = 0
  while (counter < maxIterations) {
    var sumOfSquareErrors = updateWeights(weights, values, expectedValues, thresholds);
    trainingExample(values, expectedValues);
    activateNetwork(weights, values, thresholds);
    displayNetwork(values, sumOfSquareErrors);
    counter ++;
  }

}

var numInputNodes = 2,
  numHiddenNodes = 2,
  numOutputNodes = 1,
  numNodes = numInputNodes + numHiddenNodes + numOutputNodes,
  arraySize = numNodes + 1,
  maxIterations = 40000,
  E = 2.71828,
  learningRate = 0.2,
  trainCounter = 0,
  displayCounter = 0;

function initalize(weights, values, expectedValues, thresholds) {
  for (var i = 0; i <= numNodes; i ++) {
    values[i] = 0.0;
    expectedValues[i] = 0.0;
    thresholds[i] = 0.0;
    weights[i] = [];
    for (var j = 0; j <= numNodes; j ++) {
      weights[i][j] = 0.0;
    }
  }
}

function connectNodes (weights, thresholds) {
  for (var i = 1; i <= numNodes; i ++) {
    for (var j = 1; j <= numNodes; j ++) {
      weights[i][j] = Math.random() * 2;
    }
  }

  thresholds[3] = Math.random();
  thresholds[4] = Math.random();
  thresholds[5] = Math.random();

  console.log(weights[1][3], weights[1][4], weights[2][3], weights[2][4], weights[3][5],
              weights[4][5], thresholds[3], thresholds[4], thresholds[5]);

}

function trainingExample (values, expectedValues) {

  switch (trainCounter % 4) {
    case 0:
      values[1] = 1;
      values[2] = 1;
      expectedValues[5] = 0;
      break;
    case 1:
      values[1] = 0;
      values[2] = 1;
      expectedValues[5] = 1;
      break;
    case 2:
      values[1] = 1;
      values[2] = 0;
      expectedValues[5] = 1;
      break;
    case 3:
      values[1] = 0;
      values[2] = 0;
      expectedValues[5] = 0;
      break;
  }
  trainCounter ++;
}


function activateNetwork (weights, values, thresholds) {
  // for each hidden node
  for (var i = 1 + numInputNodes; i < 1 + numInputNodes + numHiddenNodes; i ++ ) {
    var weightedInput = 0;
    // add up the weighted input
    for (var j = 1; j < 1 + numInputNodes; j ++) {
      weightedInput += weights[j][i] * values[j];
    }
    // handle thresholds

    weightedInput += (-1 * thresholds[i]);
    values[i] = 1/(1 + Math.pow(E, -weightedInput));
  }
  // for each output node
  for (var i = 1 + numInputNodes + numHiddenNodes; i < 1 + numNodes; i ++) {
    var weightedInput = 0;
    for (var j = 1 + numInputNodes; j < 1 + numInputNodes + numHiddenNodes; j ++) {
      weightedInput += weights[j][i] * values[j];

    }
    weightedInput += (-1 * thresholds[i]);
    values[i] = 1/(1 + Math.pow(E, -weightedInput));
  }
}

function updateWeights (weights, values, expectedValues, thresholds) {
  var sumOfSquareErrors = 0;
  for (var i = 1 + numInputNodes + numHiddenNodes; i < 1 + numNodes; i ++) {
    var absoluteError = expectedValues[i] - values[i];
    sumOfSquareErrors += Math.pow(absoluteError, 2);
    var outputErrorGradient = values[i] * (1 - values[i]) * absoluteError;

    for (var j = 1 + numInputNodes; j < 1 + numInputNodes + numHiddenNodes; j ++) {
      var delta = learningRate * values[j] * outputErrorGradient;
      weights[j][i] += delta;
      var hiddenErrorGradient = values[j] * (1 - values[j]) * outputErrorGradient * weights[j][i];
      
      for (var k = 1; k < 1 + numInputNodes; k ++) {
        delta = learningRate * values[k] * hiddenErrorGradient;
        weights[k][j] += delta;
      }

      var thresholdDelta = learningRate * -1 * hiddenErrorGradient;
      thresholds[j] += thresholdDelta;
    }
    var delta = learningRate * -1 * outputErrorGradient;
    thresholds[i] += delta;
  }
  return sumOfSquareErrors;
}

function displayNetwork (values, sumOfSquareErrors) {
  if (displayCounter % 4 === 0 ) {
    console.log("\n")
  }
  console.log(values[1], values[2], values[5])
  console.log("error: " + sumOfSquareErrors);
}

main();
