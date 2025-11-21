# PaddleOCR.NET

ONNX-based PaddleOCR implementation for .NET with Fluent API

## Features

- PP-OCRv5 Detection Model support
- Image preprocessing with automatic resizing (960×X or X×960)
- Fluent API for easy configuration
- SkiaSharp for high-performance image processing
- .NET 10 compatible

## Installation

This library requires:
- Microsoft.ML.OnnxRuntime (1.23.2)
- SkiaSharp (3.119.1)

## Usage

### Basic Detection

```csharp
using PaddleOCR.NET.Models;

// Create detection model
using var detector = new DetectionModelV5("detection/v5/det.onnx");

// Load image
var imageBytes = File.ReadAllBytes("document.jpg");

// Detect text regions
var result = detector.Detect(imageBytes);

// Process results
foreach (var box in result.Boxes)
{
    Console.WriteLine($"Confidence: {box.Confidence:F2}");
    foreach (var point in box.Points)
    {
        Console.WriteLine($"  Point: ({point.X:F1}, {point.Y:F1})");
    }
}
```

### Using Builder Pattern

```csharp
using PaddleOCR.NET.Models;

// Create detector with custom configuration
using var detector = new DetectionModelV5Builder()
    .WithModelPath("detection/v5/det.onnx")
    .WithTargetSize(960)       // Longer side target size
    .WithThreshold(0.3f)         // Detection threshold
    .WithBoxThreshold(0.5f)  // Box confidence threshold
    .WithUnclipRatio(1.6f)            // Box expansion ratio
    .Build();

var result = detector.Detect(imageBytes);
```

### Batch Processing

```csharp
var imageBatch = new[]
{
    File.ReadAllBytes("image1.jpg"),
    File.ReadAllBytes("image2.jpg"),
    File.ReadAllBytes("image3.jpg")
};

var results = detector.DetectBatch(imageBatch);

foreach (var result in results)
{
    Console.WriteLine($"Found {result.Boxes.Count} text regions");
}
```

## Model Download

Download PP-OCRv5 detection models from:
https://huggingface.co/monkt/paddleocr-onnx

## Architecture

The library is organized into the following modules:

- **Models**: Core detection model interfaces and implementations
- **ImageProcessing**: Image resizing and normalization utilities using SkiaSharp
- **Tensor**: Post-processing for model outputs

## Image Processing

Images are automatically:
1. Resized to 960×X or X×960 (maintaining aspect ratio)
2. Padded to multiples of 32 (PP-OCR requirement)
3. Normalized using ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

All image processing is done using SkiaSharp for optimal performance.

## License

Apache 2.0
