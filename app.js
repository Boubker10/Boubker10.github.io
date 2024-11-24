let model;
const modelPath = "assets/model.json"; // Path to TensorFlow.js model
const classes = ['carrot', 'eggplant', 'peas', 'potato', 'sweetcorn', 'tomato', 'turnip']; // Predefined classes

// Load the TensorFlow.js model
(async function loadModel() {
  try {
    console.log("Loading model...");
    model = await tf.loadGraphModel(modelPath);
    console.log("Model loaded successfully!");
  } catch (error) {
    console.error("Error loading the model:", error);
  }
})();

// Normalize the image orientation using canvas
function normalizeImage(image) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0);
  return canvas;
}

// Preprocess and make a prediction
async function makePrediction(image) {
  try {
    const tensor = tf.browser
      .fromPixels(image)
      .resizeNearestNeighbor([128, 128]) // Resize to 128x128
      .toFloat()
      .div(255.0) // Normalize to [0, 1]
      .sub([0.485, 0.456, 0.406]) // Subtract ImageNet mean
      .div([0.229, 0.224, 0.225]) // Divide by standard deviation
      .expandDims(); // Add batch dimension

    // Rearrange dimensions to NCHW (batch, channels, height, width)
    const nchwTensor = tensor.transpose([0, 3, 1, 2]);

    // Predict using the model
    const output = await model.predict(nchwTensor);
    const logits = output.dataSync(); // Extract raw logits
    const probabilities = softmax(logits); // Apply softmax
    const predictedClass = probabilities.indexOf(Math.max(...probabilities));

    console.log("Probabilities:", probabilities);
    console.log("Predicted class index:", predictedClass);

    return { probabilities, predictedClass };
  } catch (error) {
    console.error("Error during prediction:", error);
    return null;
  }
}

// Softmax function
function softmax(logits) {
  const expScores = logits.map((x) => Math.exp(x));
  const sumExpScores = expScores.reduce((a, b) => a + b, 0);
  return expScores.map((x) => x / sumExpScores);
}

// Display the predictions and probabilities
function displayPredictions(predictionData) {
  const predictionOutputElement = document.getElementById("prediction-output");
  const probabilityBarsElement = document.getElementById("probability-bars");

  // Reset the prediction areas
  predictionOutputElement.innerHTML = "";
  probabilityBarsElement.innerHTML = "";

  if (!predictionData) {
    predictionOutputElement.innerText = "Error in prediction.";
    return;
  }

  const { probabilities, predictedClass } = predictionData;

  // Show predicted class
  predictionOutputElement.innerHTML = `
    <h4>Predicted Class: <span style="color: #007bff;">${classes[predictedClass]}</span></h4>
  `;

  // Create probability bars
  probabilities.forEach((prob, i) => {
    const isMax = i === predictedClass; // Highlight the max probability
    const barWrapper = document.createElement("div");
    barWrapper.style = `
      border: ${isMax ? "2px solid #007bff" : "1px solid #007bff"};
      border-radius: 10px;
      padding: 10px;
      margin-bottom: 15px;
      background-color: #f8f9fa;
    `;

    const barContainer = document.createElement("div");
    barContainer.style = `
      display: flex;
      align-items: center;
      margin-bottom: 10px;
    `;

    // Class name
    const className = document.createElement("div");
    className.innerText = classes[i].toUpperCase();
    className.style = `
      width: 20%;
      font-weight: ${isMax ? "bold" : "normal"};
      font-size: 14px;
      color: ${isMax ? "#007bff" : "black"};
    `;

    // Probability bar background
    const barBackground = document.createElement("div");
    barBackground.style = `
      width: 60%;
      height: 15px;
      background-color: #e9ecef;
      border-radius: 7px;
      position: relative;
      overflow: hidden;
    `;

    // Probability bar fill
    const barFill = document.createElement("div");
    barFill.style = `
      width: ${Math.round(prob * 100)}%; // Display as an integer
      height: 100%;
      background-color: #007bff;
      position: absolute;
      border-radius: 7px;
      transition: width 0.5s ease-in-out;
    `;

    barBackground.appendChild(barFill);

    // Probability percentage
    const probPercent = document.createElement("div");
    probPercent.innerText = `${Math.round(prob * 100)}%`; // Convert to integer
    probPercent.style = `
      width: 20%;
      text-align: right;
      font-size: 14px;
      font-weight: ${isMax ? "bold" : "normal"};
      color: ${isMax ? "#007bff" : "black"};
    `;

    barContainer.appendChild(className);
    barContainer.appendChild(barBackground);
    barContainer.appendChild(probPercent);
    barWrapper.appendChild(barContainer);
    probabilityBarsElement.appendChild(barWrapper);
  });
}

// Image upload and drag-and-drop functionality
document.getElementById("image-upload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const supportedFormats = ["image/jpeg", "image/png"];
  if (!supportedFormats.includes(file.type)) {
    alert("Unsupported image format. Please upload a JPEG or PNG file.");
    return;
  }

  const img = new Image();
  img.crossOrigin = "anonymous"; // Handle CORS
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    const normalizedImg = normalizeImage(img); // Normalize orientation
    document.getElementById("output-image-upload").innerHTML = `
      <img src="${img.src}" alt="Uploaded Image" style="max-width: 100%; border-radius: 10px;">
    `;
    const predictions = await makePrediction(normalizedImg);
    displayPredictions(predictions);
  };
});
