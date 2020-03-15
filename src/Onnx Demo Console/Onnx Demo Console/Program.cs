using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;

namespace Onnx_Demo_Console
{
    public class Score
    {
        public int Prediction { get; set; }
        public double Time { get; set; }
        public List<double> Scores { get; set; }
        public string Status { get; set; }
        public bool Empty { get; set; } = true;
    }

    class Program
    {
        static void Main(string[] args)
        {
            var session = new InferenceSession(@"model\model.onnx");
            var img = Image.FromFile(@"input\sample_image.png");
            var bmp = ResizeImage(img, 28, 28);
            var data = ConvertImageToByteArray(bmp);
            var result = PredictLocal(session, data);
            Console.WriteLine(result);
        }

        public static float[] ConvertImageToByteArray(Bitmap image)
        {
            var imageByteArray = new float[image.Width * image.Height] ;

            for (int x = 0; x < image.Width; x++)
            {
                for (int y = 0; y < image.Height; y++)
                {
                    var p = image.GetPixel(x, y);
                    imageByteArray[y * image.Width + x] = 255 - (((float)(p.R) + p.G + p.B) / 3);
                }
            }

            return imageByteArray;
        }

        static Score PredictLocal(InferenceSession session, float[] digit)
        {
            var now = DateTime.Now;
            Tensor<float> x = new DenseTensor<float>(digit.Length);

            for (int i = 0; i <= digit.Length - 1; i++)
                x[i] = digit[i] / 255.0f;

            var input = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor("0", x)
            };

            try
            {
                var prediction = session.Run(input).First().AsTensor<float>().ToArray();
                return new Score()
                {
                    Status = $"Local Mode: {session}",
                    Empty = false,
                    Prediction = Array.IndexOf(prediction, prediction.Max()),
                    Scores = prediction.Select(i => System.Convert.ToDouble(i)).ToList(),
                    Time = (DateTime.Now - now).TotalSeconds
                };
            }
            catch (Exception e)
            {
                return new Score()
                {
                    Status = e.Message
                };
            }
        }

        /// <summary>
        /// Resize the image to the specified width and height.
        /// </summary>
        /// <param name="image">The image to resize.</param>
        /// <param name="width">The width to resize to.</param>
        /// <param name="height">The height to resize to.</param>
        /// <returns>The resized image.</returns>
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
    }
}
