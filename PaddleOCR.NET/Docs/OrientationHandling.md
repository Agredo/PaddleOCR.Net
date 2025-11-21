# Image Orientation Handling

## Overview

The PaddleOCR.NET library now automatically handles EXIF orientation metadata in images. This means that images taken with smartphones or cameras that have orientation information will be automatically rotated to the correct orientation before processing.

## Why This Matters

Many images, especially those taken with smartphones, contain EXIF orientation metadata that indicates how the image should be displayed. Without proper handling:

- **Portrait photos** might be processed as landscape
- **Rotated images** might not detect text correctly
- **Bounding boxes** would be in the wrong positions

## How It Works

### Automatic Orientation Correction

All image loading now uses `ImageLoader.LoadWithOrientation()` which:

1. Reads the EXIF orientation tag from the image
2. Applies the necessary transformation (rotation/flip)
3. Returns a correctly oriented bitmap

### Supported EXIF Orientations

The library supports all 8 EXIF orientation values:

| EXIF Value | Orientation | Transformation |
|------------|-------------|----------------|
| 1 | TopLeft | None (normal) |
| 2 | TopRight | Flip horizontal |
| 3 | BottomRight | Rotate 180° |
| 4 | BottomLeft | Flip vertical |
| 5 | LeftTop | Rotate 90° CCW + Flip horizontal |
| 6 | RightTop | Rotate 90° CW |
| 7 | RightBottom | Rotate 90° CW + Flip horizontal |
| 8 | LeftBottom | Rotate 90° CCW |

## Usage

### Basic Detection (Automatic)

```csharp
using var detector = new DetectionModelV5("path/to/det.onnx");
var imageBytes = File.ReadAllBytes("photo.jpg");

// Orientation is automatically corrected
var result = detector.Detect(imageBytes);

Console.WriteLine($"Detected {result.Boxes.Count} text regions");
```

### Manual Image Loading

```csharp
// Load from file path
using var bitmap = ImageLoader.LoadWithOrientation("path/to/image.jpg");

// Load from byte array
var imageBytes = File.ReadAllBytes("path/to/image.jpg");
using var bitmap2 = ImageLoader.LoadWithOrientation(imageBytes);

// Load from stream
using var stream = File.OpenRead("path/to/image.jpg");
using var bitmap3 = ImageLoader.LoadWithOrientation(stream);
```

### Exporting Results

```csharp
// Export with boxes - the output image will be in correct orientation
var outputPath = ImageExporter.ExportWithBoxes(imagePath, result);
```

## Diagnostic Output

The `ImageLoader` provides console output for debugging:

```
[IMAGE-LOADER] Detected EXIF orientation: RightTop
[IMAGE-LOADER] Applying orientation correction: RightTop -> TopLeft
```

If no orientation correction is needed:

```
[IMAGE-LOADER] Detected EXIF orientation: TopLeft
[IMAGE-LOADER] No orientation correction needed
```

## Common Scenarios

### Smartphone Photos

Most smartphone photos contain EXIF orientation data:
- Photos taken in portrait mode: Usually `RightTop` (90° CW rotation needed)
- Photos taken in landscape mode: Usually `TopLeft` (no rotation)
- Photos taken upside-down: Usually `LeftBottom` (90° CCW rotation needed)

### Scanned Documents

Scanned documents typically don't have EXIF orientation data, so no correction is applied.

### Screenshots

Screenshots usually don't have EXIF orientation data and are already correctly oriented.

## Performance Impact

The orientation correction adds minimal overhead:
- **Reading EXIF data**: ~1-5ms
- **Applying transformation**: ~10-50ms (depends on image size)

The correction is only applied when needed, so images without orientation metadata have no performance impact.

## Troubleshooting

### Problem: Text not detected in portrait photos

**Solution**: This is now automatically handled. If you're still experiencing issues, check:

1. Verify the image has EXIF data: Look for console output `[IMAGE-LOADER] Detected EXIF orientation:`
2. Check if the model threshold is appropriate for your image
3. Verify the image quality and resolution

### Problem: Bounding boxes in wrong positions

**Solution**: This was likely caused by orientation issues and should now be fixed. If persisting:

1. Check console output to confirm orientation was detected
2. Verify you're using the latest version that includes `ImageLoader`
3. Export the result image to visually verify box positions

### Problem: Image dimensions seem swapped

**Solution**: This indicates an orientation issue that should now be fixed. For 90°/270° rotations, width and height are swapped, which is expected and correct.

## Migration from Previous Versions

If you were manually handling rotation in your code:

**Before:**
```csharp
// Manual rotation was needed
using var bitmap = SKBitmap.Decode(imageData);
// ... manual rotation code ...
```

**Now:**
```csharp
// Automatic rotation
using var detector = new DetectionModelV5("path/to/det.onnx");
var result = detector.Detect(imageData); // Orientation handled automatically
```

You can safely remove any manual rotation code you had implemented.

## Technical Details

### Implementation

The orientation correction uses:
- `SKCodec` to read the `EncodedOrigin` property
- `SKCanvas` transformations (rotate, scale, translate) to apply corrections
- Proper dimension swapping for 90°/270° rotations

### Tested Image Formats

Orientation correction is supported for formats that can contain EXIF data:
- ? JPEG/JPG
- ? TIFF
- ? WebP (if contains EXIF)
- ?? PNG (rare, but supported if present)
- ? BMP (no EXIF support)
- ? GIF (no EXIF support)

## See Also

- [DetectionExample.cs](DetectionExample.cs) - Basic detection examples
- [OrientationExample.cs](OrientationExample.cs) - Orientation-specific examples
- [ImageExporter.cs](../ImageProcessing/ImageExporter.cs) - Exporting results
