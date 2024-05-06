// Import necessary libraries
const express = require("express");
const http = require("http");
const WebSocket = require("ws");
const bodyParser = require("body-parser");
const cors = require("cors");
const morgan = require("morgan");
const { spawn } = require("child_process");

// Create Express app
const app = express();
app.use(cors());
app.use(morgan("dev"));
const port = 3000;

// Create HTTP server
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocket.Server({ server });

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// WebSocket connection handler
wss.on("connection", (ws) => {
  console.log("Client connected");

  // Handle messages received from the client
  ws.on("message", (message) => {
    console.log("Received message:", message);
    // You can process the received message here if needed
  });

  // Handle WebSocket connection closure
  ws.on("close", () => {
    console.log("Client disconnected");
  });
});

// Endpoint to receive video frames from frontend
app.post("/video-frames", async (req, res) => {
  try {
    // Assuming the video frames are sent as base64 encoded strings in the request body
    const frames = req.body.frames;
    console.log("flag");

    // Pass frames to machine learning model for processing
    const mlOutput = await processWithMLModel(frames);

    console.log(mlOutput);
    // Send machine learning model output back to frontend
    res.json({ mlOutput });
  } catch (error) {
    return res.status(500).json(error);
  }
});

async function processWithMLModel(frames) {
  // Process frames with your machine learning model
  // Replace this with your actual machine learning code
  // Here, we'll just simulate calling a Python script with frames as arguments
  const pythonProcess = spawn("python", [
    "D://Study/ML Test/test.py",
    "frames",
  ]);
  return new Promise((resolve, reject) => {
    // Handle stdout from the Python process
    pythonProcess.stdout.on("data", (data) => {
      const mlOutput = data.toString().trim(); // Assuming the output is a string
      resolve(mlOutput);
    });

    // Handle any errors from the Python process
    pythonProcess.on("error", (error) => {
      console.error(`Error executing Python script: ${error.message}`);
      reject(error);
    });
  });
}

// Start the server
server.listen(port, () => {
  console.log(`Server is listening at http://localhost:${port}`);
});
