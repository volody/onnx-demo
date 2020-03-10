using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
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
            var img = ConvertImageToByteArray(@"input\sample_image.png");
            var result = PredictLocal(session, img);
            Console.WriteLine(result);
        }

        public static float[] ConvertImageToByteArray(string imagePath)
        {
            float[] imageByteArray = null;
            FileStream fileStream = new FileStream(imagePath, FileMode.Open, FileAccess.Read);
            using (BinaryReader reader = new BinaryReader(fileStream))
            {
                imageByteArray = new float[reader.BaseStream.Length];
                for (int i = 0; i < reader.BaseStream.Length; i++)
                    imageByteArray[i] = reader.ReadByte();
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

    }
}
