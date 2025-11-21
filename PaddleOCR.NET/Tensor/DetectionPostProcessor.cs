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
    /// <returns>List of bounding boxes</returns>
    public static List<BoundingBox> ExtractBoxes(
        float[] output,
        int paddedWidth,
        int paddedHeight,
        int resizedWidth,
        int resizedHeight,
        (int Width, int Height) originalSize,
        float threshold = 0.3f,
        float boxThreshold = 0.5f)
    {
        var boxes = new List<BoundingBox>();

        // Output is a probability map with shape [1, 1, H, W]
        // Detection models typically downsample by 4x
        var mapHeight = paddedHeight / 4;
        var mapWidth = paddedWidth / 4;

        // Apply threshold and find contours
        var binaryMap = new bool[mapHeight, mapWidth];
        for (int y = 0; y < mapHeight; y++)
        {
            for (int x = 0; x < mapWidth; x++)
            {
                var idx = y * mapWidth + x;
                binaryMap[y, x] = idx < output.Length && output[idx] > threshold;
            }
        }

        // Extract contours
        var contours = ExtractContours(binaryMap);
        
        // Calculate scaling factors
        var scaleX = (float)originalSize.Width / resizedWidth;
        var scaleY = (float)originalSize.Height / resizedHeight;

        foreach (var contour in contours)
        {
            if (contour.Count < 4) continue;

            // Calculate bounding box
            var minX = contour.Min(p => p.X);
            var maxX = contour.Max(p => p.X);
            var minY = contour.Min(p => p.Y);
            var maxY = contour.Max(p => p.Y);

            // Scale back to original image size
            var points = new[]
            {
                new[] { minX * 4 * scaleX, minY * 4 * scaleY },
                new[] { maxX * 4 * scaleX, minY * 4 * scaleY },
                new[] { maxX * 4 * scaleX, maxY * 4 * scaleY },
                new[] { minX * 4 * scaleX, maxY * 4 * scaleY }
            };

            var confidence = CalculateConfidence(output, contour, mapWidth);
            
            if (confidence > boxThreshold)
            {
                boxes.Add(new BoundingBox(points, confidence));
            }
        }

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
