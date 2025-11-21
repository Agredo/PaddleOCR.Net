using PaddleOCR.NET.Models.Detection;

namespace PaddleOCR.NET.Tensor;

/// <summary>
/// Post-processing utilities for detection model output
/// </summary>
public static class DetectionPostProcessor
{
    /// <summary>
    /// Extracts bounding boxes from detection model output
    /// </summary>
    /// <param name="output">Model output probability map</param>
    /// <param name="paddedWidth">Padded image width</param>
    /// <param name="paddedHeight">Padded image height</param>
    /// <param name="resizedWidth">Resized image width (before padding)</param>
    /// <param name="resizedHeight">Resized image height (before padding)</param>
    /// <param name="originalSize">Original image size</param>
    /// <param name="threshold">Detection threshold (default: 0.3)</param>
    /// <param name="boxThreshold">Box confidence threshold (default: 0.5)</param>
    /// <param name="unclipRatio">Ratio for expanding boxes (default: 1.6)</param>
    /// <returns>List of bounding boxes</returns>
    public static List<BoundingBox> ExtractBoxes(
        float[] output,
        int paddedWidth,
        int paddedHeight,
        int resizedWidth,
        int resizedHeight,
        (int Width, int Height) originalSize,
        float threshold = 0.3f,
        float boxThreshold = 0.5f,
        float unclipRatio = 1.6f)
    {
        var boxes = new List<BoundingBox>();

        // Output is a probability map - typically downsampled 4x
        var mapHeight = paddedHeight / 4;
        var mapWidth = paddedWidth / 4;

        Console.WriteLine($"[POST-PROCESS] Starting box extraction:");
        Console.WriteLine($"  Map dimensions: {mapWidth}x{mapHeight}");
        Console.WriteLine($"  Thresholds: detection={threshold}, box={boxThreshold}");
        Console.WriteLine($"  Unclip ratio: {unclipRatio}");

        // Verify output dimensions match expectations
        if (output.Length != mapWidth * mapHeight)
        {
            Console.WriteLine($"[WARNING] Output length mismatch!");
            Console.WriteLine($"  Expected: {mapWidth * mapHeight}, Got: {output.Length}");
            
            // Try to infer actual dimensions
            var possibleWidth = (int)Math.Sqrt(output.Length);
            if (possibleWidth * possibleWidth == output.Length)
            {
                mapWidth = possibleWidth;
                mapHeight = possibleWidth;
                Console.WriteLine($"[INFO] Adjusted to square: {mapWidth}x{mapHeight}");
            }
            else
            {
                // Try different aspect ratios
                for (int w = 1; w <= output.Length; w++)
                {
                    if (output.Length % w == 0)
                    {
                        int h = output.Length / w;
                        if (Math.Abs((float)w / h - (float)paddedWidth / paddedHeight) < 0.1f)
                        {
                            mapWidth = w;
                            mapHeight = h;
                            Console.WriteLine($"[INFO] Adjusted to: {mapWidth}x{mapHeight}");
                            break;
                        }
                    }
                }
            }
        }

        // Create binary map
        var binaryMap = new bool[mapHeight, mapWidth];
        var pixelsAboveThreshold = 0;
        
        for (int y = 0; y < mapHeight; y++)
        {
            for (int x = 0; x < mapWidth; x++)
            {
                var idx = y * mapWidth + x;
                if (idx < output.Length && output[idx] > threshold)
                {
                    binaryMap[y, x] = true;
                    pixelsAboveThreshold++;
                }
            }
        }

        Console.WriteLine($"  Binary map: {pixelsAboveThreshold} pixels above threshold ({100.0 * pixelsAboveThreshold / (mapWidth * mapHeight):F2}%)");

        if (pixelsAboveThreshold == 0)
        {
            Console.WriteLine("[WARNING] No pixels above threshold! Try lowering the threshold value.");
            return boxes;
        }

        // Extract contours
        var contours = ExtractContours(binaryMap);
        Console.WriteLine($"  Found {contours.Count} contours");

        // Calculate direct scaling from map space to original image space
        // The model downsamples by 4x, so map dimensions are paddedWidth/4 and paddedHeight/4
        // We need to scale directly from map coordinates to original image coordinates
        var mapToOriginalScaleX = (float)originalSize.Width / mapWidth;
        var mapToOriginalScaleY = (float)originalSize.Height / mapHeight;

        Console.WriteLine($"  Scaling: mapToOriginal = ({mapToOriginalScaleX:F3}, {mapToOriginalScaleY:F3})");

        int boxIndex = 0;
        foreach (var contour in contours)
        {
            if (contour.Count < 4)
            {
                Console.WriteLine($"  Contour {boxIndex}: {contour.Count} points - SKIPPED (too few points)");
                boxIndex++;
                continue;
            }

            // Get bounding rectangle in map space
            var minX = contour.Min(p => p.X);
            var maxX = contour.Max(p => p.X);
            var minY = contour.Min(p => p.Y);
            var maxY = contour.Max(p => p.Y);
            
            var mapWidth_box = maxX - minX + 1;
            var mapHeight_box = maxY - minY + 1;

            // Filter very small detections (likely noise)
            if (mapWidth_box < 2 || mapHeight_box < 2)
            {
                Console.WriteLine($"  Contour {boxIndex}: {mapWidth_box}x{mapHeight_box} - SKIPPED (too small)");
                boxIndex++;
                continue;
            }

            // Calculate confidence
            var confidence = CalculateConfidence(output, contour, mapWidth);
            
            // Scale directly from map space to original image coordinates
            var originalMinX = minX * mapToOriginalScaleX;
            var originalMaxX = maxX * mapToOriginalScaleX;
            var originalMinY = minY * mapToOriginalScaleY;
            var originalMaxY = maxY * mapToOriginalScaleY;

            // Calculate center point and current dimensions
            var centerX = (originalMinX + originalMaxX) / 2f;
            var centerY = (originalMinY + originalMaxY) / 2f;
            var currentWidth = originalMaxX - originalMinX;
            var currentHeight = originalMaxY - originalMinY;

            // Expand box using unclipRatio
            var newWidth = currentWidth * unclipRatio;
            var newHeight = currentHeight * unclipRatio;

            // Calculate new corners around center point
            var expandedMinX = centerX - newWidth / 2f;
            var expandedMaxX = centerX + newWidth / 2f;
            var expandedMinY = centerY - newHeight / 2f;
            var expandedMaxY = centerY + newHeight / 2f;

            // Clamp to image bounds
            expandedMinX = Math.Max(0, expandedMinX);
            expandedMaxX = Math.Min(originalSize.Width, expandedMaxX);
            expandedMinY = Math.Max(0, expandedMinY);
            expandedMaxY = Math.Min(originalSize.Height, expandedMaxY);

            var points = new[]
            {
                new[] { expandedMinX, expandedMinY },
                new[] { expandedMaxX, expandedMinY },
                new[] { expandedMaxX, expandedMaxY },
                new[] { expandedMinX, expandedMaxY }
            };

            Console.WriteLine($"  Contour {boxIndex}: {contour.Count} pts, map=({minX},{minY})-({maxX},{maxY}), conf={confidence:F3}");
            Console.WriteLine($"    -> Original scaled: ({originalMinX:F1},{originalMinY:F1})-({originalMaxX:F1},{originalMaxY:F1})");
            Console.WriteLine($"    -> Expanded ({unclipRatio}x): ({expandedMinX:F1},{expandedMinY:F1})-({expandedMaxX:F1},{expandedMaxY:F1})");

            if (confidence >= boxThreshold)
            {
                boxes.Add(new BoundingBox(points, confidence));
                Console.WriteLine($"    -> ADDED (confidence {confidence:F3} >= {boxThreshold})");
            }
            else
            {
                Console.WriteLine($"    -> SKIPPED (confidence {confidence:F3} < {boxThreshold})");
            }
            
            boxIndex++;
        }

        Console.WriteLine($"[POST-PROCESS] Extracted {boxes.Count} boxes from {contours.Count} contours");
        return boxes;
    }

    /// <summary>
    /// Extracts contours from a binary map using flood fill
    /// </summary>
    private static List<List<(int X, int Y)>> ExtractContours(bool[,] binaryMap)
    {
        var height = binaryMap.GetLength(0);
        var width = binaryMap.GetLength(1);
        var visited = new bool[height, width];
        var contours = new List<List<(int X, int Y)>>();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (binaryMap[y, x] && !visited[y, x])
                {
                    var contour = new List<(int X, int Y)>();
                    FloodFill(binaryMap, visited, x, y, contour);
                    
                    if (contour.Count >= 4)
                        contours.Add(contour);
                }
            }
        }

        return contours;
    }

    /// <summary>
    /// Performs flood fill to extract a contour
    /// </summary>
    private static void FloodFill(
        bool[,] map, 
        bool[,] visited, 
        int startX, 
        int startY, 
        List<(int X, int Y)> contour)
    {
        var height = map.GetLength(0);
        var width = map.GetLength(1);
        var stack = new Stack<(int X, int Y)>();
        stack.Push((startX, startY));

        while (stack.Count > 0)
        {
            var (x, y) = stack.Pop();
            
            if (x < 0 || x >= width || y < 0 || y >= height || visited[y, x] || !map[y, x])
                continue;

            visited[y, x] = true;
            contour.Add((x, y));

            // 8-neighborhood
            stack.Push((x + 1, y));
            stack.Push((x - 1, y));
            stack.Push((x, y + 1));
            stack.Push((x, y - 1));
            stack.Push((x + 1, y + 1));
            stack.Push((x + 1, y - 1));
            stack.Push((x - 1, y + 1));
            stack.Push((x - 1, y - 1));
        }
    }

    /// <summary>
    /// Calculates average confidence for a contour
    /// </summary>
    private static float CalculateConfidence(
        float[] output, 
        List<(int X, int Y)> contour, 
        int mapWidth)
    {
        if (contour.Count == 0) return 0f;

        var sum = 0f;
        foreach (var (x, y) in contour)
        {
            var idx = y * mapWidth + x;
            if (idx < output.Length)
                sum += output[idx];
        }

        return sum / contour.Count;
    }
}
