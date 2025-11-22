# Complete OCR Pipeline - Combining Detection and Recognition

This guide shows you how to combine the Detection and Recognition models to create a complete end-to-end OCR solution.

## Table of Contents
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Basic Usage](#basic-usage)
- [Advanced Examples](#advanced-examples)
- [Image Cropping](#image-cropping)
- [Pipeline Builder](#pipeline-builder)
- [Best Practices](#best-practices)

## Quick Start

### Simple OCR Pipeline

```csharp
using PaddleOCR.NET.Pipeline;

// Build the pipeline
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(builder => builder
        .WithModelPath("path/to/det.onnx")
        .WithThreshold(0.3f)
        .WithBoxThreshold(0.5f))
    .WithRecognition(builder => builder
        .WithModelPath("path/to/rec.onnx")
        .WithCharacterDict("path/to/dict.txt")
        .WithBatchSize(6))
    .Build();

// Process an image
var imageBytes = File.ReadAllBytes("document.jpg");
var result = pipeline.Process(imageBytes);

// Display results
foreach (var region in result.TextRegions)
{
    Console.WriteLine($"{region.Text.Text} (confidence: {region.Text.Confidence:F3})");
}

// Get all text
Console.WriteLine(result.GetFullText());
```

## Architecture

### Component Overview

```
???????????????????????????????????????????????????????
?                   OCRPipeline                       ?
???????????????????????????????????????????????????????
?                                                     ?
?  ????????????????      ???????????????           ?
?  ?  Detection   ? ???> ? Image       ?           ?
?  ?  Model       ?      ? Cropper     ?           ?
?  ????????????????      ???????????????           ?
?         ?                      ?                  ?
?         ?                      ?                  ?
?         ?              ???????????????           ?
?         ?              ?Recognition  ?           ?
?         ?              ?Model (Batch)?           ?
?         ?              ???????????????           ?
?         ?                      ?                  ?
?         ?                      ?                  ?
?    ???????????????????????????????              ?
?    ?      OCRResult              ?              ?
?    ?  (Detection + Recognition)  ?              ?
?    ???????????????????????????????              ?
?                                                   ?
?????????????????????????????????????????????????????
```

### Key Components

1. **OCRPipeline**: Orchestrates the entire OCR process
2. **ImageCropper**: Extracts text regions from images based on bounding boxes
3. **OCRResult**: Contains both detection and recognition results
4. **OCRTextRegion**: Represents a single text region with location and recognized text
5. **OCRPipelineBuilder**: Fluent builder for easy pipeline configuration

## Basic Usage

### Method 1: Using Pipeline Builder (Recommended)

```csharp
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(builder => builder
        .WithModelPath("det.onnx")
        .WithThreshold(0.3f)
        .WithBoxThreshold(0.5f)
        .WithUnclipRatio(1.6f))
    .WithRecognition(builder => builder
        .WithModelPath("rec.onnx")
        .WithCharacterDict("dict.txt")
        .WithBatchSize(6))
    .Build();

var result = pipeline.Process(imageBytes);
```

### Method 2: Manual Construction

```csharp
// Create models separately
using var detector = new DetectionModelV5Builder()
    .WithModelPath("det.onnx")
    .WithThreshold(0.3f)
    .Build();

using var recognizer = new RecognitionModelV5Builder()
    .WithModelPath("rec.onnx")
    .WithCharacterDict("dict.txt")
    .Build();

// Create pipeline
using var pipeline = new OCRPipeline(detector, recognizer);

// Process
var result = pipeline.Process(imageBytes);
```

## Advanced Examples

### Example 1: Processing Multiple Images

```csharp
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(builder => builder
        .WithModelPath("det.onnx"))
    .WithRecognition(builder => builder
        .WithModelPath("rec.onnx")
        .WithCharacterDict("dict.txt")
        .WithBatchSize(8)) // Larger batch for efficiency
    .Build();

// Process multiple images
var imageFiles = Directory.GetFiles("documents/", "*.jpg");
var imageBatch = imageFiles.Select(File.ReadAllBytes).ToArray();
var results = pipeline.ProcessBatch(imageBatch);

foreach (var result in results)
{
    Console.WriteLine($"Found {result.Count} text regions");
    Console.WriteLine(result.GetFullText());
}
```

### Example 2: Filtering and Sorting Results

```csharp
var result = pipeline.Process(imageBytes);

// Filter by confidence
var highConfidence = result.TextRegions
    .Where(r => r.Text.Confidence > 0.8f)
    .ToList();

// Sort by position (top to bottom, left to right)
var sortedRegions = result.TextRegions
    .OrderBy(r => r.BoundingBox.Points[0].Y)  // Y coordinate (top to bottom)
    .ThenBy(r => r.BoundingBox.Points[0].X)   // X coordinate (left to right)
    .ToList();

// Get text in reading order
var orderedText = string.Join("\n", sortedRegions.Select(r => r.Text.Text));
```

### Example 3: Box Merging for Better Line Detection

```csharp
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(builder => builder
        .WithModelPath("det.onnx")
        .WithThreshold(0.3f)
        .WithBoxMerging(true)                    // Enable merging
        .WithMergeDistanceThreshold(0.5f)        // Merge nearby boxes
        .WithMergeOverlapThreshold(0.1f))        // Merge overlapping boxes
    .WithRecognition(builder => builder
        .WithModelPath("rec.onnx")
        .WithCharacterDict("dict.txt"))
    .Build();

var result = pipeline.Process(imageBytes);
// Result will have merged text lines instead of individual words/characters
```

### Example 4: Detailed Analysis

```csharp
var result = pipeline.Process(imageBytes);

Console.WriteLine($"=== OCR Analysis ===");
Console.WriteLine($"Image Size: {result.DetectionResult.OriginalImageSize}");
Console.WriteLine($"Total Regions: {result.Count}");
Console.WriteLine($"Average Detection Confidence: {result.TextRegions.Average(r => r.BoundingBox.Confidence):F3}");
Console.WriteLine($"Average Recognition Confidence: {result.TextRegions.Average(r => r.Text.Confidence):F3}");
Console.WriteLine($"Total Characters: {result.TextRegions.Sum(r => r.Text.Text.Length)}");

// Find regions with low confidence
var lowConfidence = result.TextRegions
    .Where(r => r.Text.Confidence < 0.7f)
    .ToList();

if (lowConfidence.Any())
{
    Console.WriteLine($"\n?? Low confidence regions: {lowConfidence.Count}");
    foreach (var region in lowConfidence)
    {
        Console.WriteLine($"  \"{region.Text.Text}\" (confidence: {region.Text.Confidence:F3})");
    }
}
```

### Example 5: Per-Character Analysis

```csharp
var result = pipeline.Process(imageBytes);

foreach (var region in result.TextRegions)
{
    Console.WriteLine($"\nText: \"{region.Text.Text}\"");
    
    if (region.Text.CharConfidences != null)
    {
        Console.WriteLine("Character-level confidences:");
        for (int i = 0; i < region.Text.Text.Length; i++)
        {
            var confidence = region.Text.CharConfidences[i];
            var symbol = confidence > 0.8f ? "?" : confidence > 0.6f ? "~" : "?";
            Console.WriteLine($"  {symbol} '{region.Text.Text[i]}': {confidence:F3}");
        }
    }
}
```

### Example 6: Export Results to Multiple Formats

```csharp
var result = pipeline.Process(imageBytes);

// 1. Export as plain text
File.WriteAllText("output.txt", result.GetFullText());

// 2. Export as JSON
var json = System.Text.Json.JsonSerializer.Serialize(new
{
    ImageSize = result.DetectionResult.OriginalImageSize,
    TextRegions = result.TextRegions.Select(r => new
    {
        Text = r.Text.Text,
        Confidence = r.Text.Confidence,
        Position = r.BoundingBox.Points[0]
    })
}, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
File.WriteAllText("output.json", json);

// 3. Export as CSV
var csv = new List<string> { "Index,Text,Confidence,X,Y" };
csv.AddRange(result.TextRegions.Select(r =>
    $"{r.Index},\"{r.Text.Text}\",{r.Text.Confidence:F3}," +
    $"{r.BoundingBox.Points[0].X:F0},{r.BoundingBox.Points[0].Y:F0}"));
File.WriteAllLines("output.csv", csv);

// 4. Export annotated image
var outputPath = ImageExporter.ExportWithBoxes(
    "original.jpg",
    result.DetectionResult,
    outputSuffix: "_ocr",
    showConfidence: true
);
```

## Image Cropping

The `ImageCropper` class extracts text regions from images based on bounding boxes.

### Basic Cropping

```csharp
using var bitmap = ImageLoader.LoadWithOrientation(imageBytes);

// Crop a single region
var croppedBitmap = ImageCropper.CropRegion(bitmap, boundingBox);

// Crop multiple regions
var croppedBitmaps = ImageCropper.CropRegions(bitmap, boundingBoxes);
```

### Cropping from Bytes

```csharp
// Returns cropped image as PNG bytes
var croppedBytes = ImageCropper.CropRegion(imageBytes, boundingBox);
```

## Pipeline Builder

### Full Configuration Example

```csharp
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(builder => builder
        .WithModelPath("det.onnx")
        .WithTargetSize(960)              // Image preprocessing size
        .WithThreshold(0.3f)              // Detection threshold
        .WithBoxThreshold(0.5f)           // Box confidence threshold
        .WithUnclipRatio(1.6f)            // Box expansion ratio
        .WithBoxMerging(true)             // Enable box merging
        .WithMergeDistanceThreshold(0.5f) // Distance threshold for merging
        .WithMergeOverlapThreshold(0.1f)) // Overlap threshold for merging
    .WithRecognition(builder => builder
        .WithModelPath("rec.onnx")
        .WithCharacterDict("dict.txt")
        .WithBatchSize(6))                // Recognition batch size
    .Build();
```

## OCRResult API

### Properties

```csharp
public class OCRResult
{
    // List of text regions
    public IReadOnlyList<OCRTextRegion> TextRegions { get; }
    
    // Original detection result
    public DetectionResult DetectionResult { get; }
    
    // Number of text regions
    public int Count { get; }
}
```

### Methods

```csharp
// Get all text with custom separator
string fullText = result.GetFullText("\n");     // Line breaks
string fullText = result.GetFullText(" ");      // Spaces
string fullText = result.GetFullText();         // Default: newline
```

### OCRTextRegion Properties

```csharp
public class OCRTextRegion
{
    // Bounding box location
    public BoundingBox BoundingBox { get; }
    
    // Recognized text with confidence
    public RecognizedText Text { get; }
    
    // Index in detection results
    public int Index { get; }
}
```

## Best Practices

### 1. Model Selection

**Detection Model Settings:**
- Lower thresholds (0.2-0.3) for documents with faint text
- Higher thresholds (0.5-0.7) for clear, high-contrast text
- Enable box merging for text lines, disable for individual words

**Recognition Batch Size:**
- Small images: 8-16
- Medium images: 4-8
- Large images: 2-4

### 2. Performance Optimization

```csharp
// ? Good: Reuse pipeline for multiple images
using var pipeline = BuildPipeline();
foreach (var image in images)
{
    var result = pipeline.Process(image);
}

// ? Bad: Creating new pipeline each time
foreach (var image in images)
{
    using var pipeline = BuildPipeline(); // Wasteful!
    var result = pipeline.Process(image);
}
```

### 3. Error Handling

```csharp
try
{
    var result = pipeline.Process(imageBytes);
    
    if (result.Count == 0)
    {
        Console.WriteLine("No text detected");
        return;
    }
    
    // Process results
}
catch (FileNotFoundException ex)
{
    Console.WriteLine($"Model file not found: {ex.Message}");
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"Failed to process image: {ex.Message}");
}
```

### 4. Result Validation

```csharp
var validRegions = result.TextRegions
    .Where(r => r.Text.Confidence > 0.7f)           // Good recognition
    .Where(r => r.BoundingBox.Confidence > 0.5f)    // Good detection
    .Where(r => !string.IsNullOrWhiteSpace(r.Text.Text)) // Not empty
    .ToList();
```

### 5. Memory Management

```csharp
// Dispose pipeline when done (disposes both models)
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(...)
    .WithRecognition(...)
    .Build();

// Pipeline automatically disposed at end of using block
```

## Performance Tips

1. **Batch Processing**: Process multiple images to amortize model loading overhead
2. **Appropriate Batch Size**: Balance between memory usage and throughput
3. **Box Merging**: Enable for documents, disable for individual word recognition
4. **Image Size**: Larger detection target size = better accuracy but slower
5. **Confidence Thresholds**: Filter low-confidence results to reduce false positives

## Common Use Cases

### Document Scanning
```csharp
.WithDetection(builder => builder
    .WithThreshold(0.3f)
    .WithBoxMerging(true)           // Merge into lines
    .WithMergeDistanceThreshold(0.5f))
```

### Scene Text (Photos)
```csharp
.WithDetection(builder => builder
    .WithThreshold(0.5f)            // Higher threshold for noise
    .WithBoxMerging(false))         // Keep individual words
```

### Receipt/Invoice Processing
```csharp
.WithDetection(builder => builder
    .WithThreshold(0.3f)
    .WithTargetSize(1280))          // Larger size for small text
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No text detected | Lower detection threshold, verify model path |
| Wrong text recognized | Check character dictionary matches language |
| Low confidence | Improve image quality, adjust thresholds |
| Slow processing | Reduce image size, increase batch size |
| Out of memory | Decrease batch size, process fewer images at once |

## Next Steps

- See `Examples/OCRPipelineExample.cs` for more examples
- Read `Models/Detection/README.md` for detection details
- Read `Models/Recognition/README.md` for recognition details
- Check `ImageProcessing/ImageCropper.cs` for cropping API
