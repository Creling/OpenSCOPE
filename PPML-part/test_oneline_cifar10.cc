#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

vector<double> read_file(const std::string &filename)
{

    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file" << std::endl;
        return {};
    }

    std::vector<double> weights;
    std::string line;

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        double value;

        while (iss >> value)
        {
            weights.push_back(value);
        }
    }

    file.close();

    std::cout << "Read " << weights.size() << " weights:" << std::endl;

    return weights;
}

vector<double> read_image(const char *filename)
{
    int width = 32;
    int height = 32;
    int channels = 3;
    unsigned char *image_data = stbi_load(filename, &width, &height, &channels, 0);

    if (!image_data)
    {
        cerr << "Could not load the image in " << filename << endl;
        return vector<double>();
    }

    vector<double> imageVector;
    imageVector.reserve(width * height * channels);

    for (int i = 0; i < width * height; ++i)
    {
        // Channel R
        imageVector.push_back(static_cast<double>(image_data[3 * i]) / 255.0f);
    }
    for (int i = 0; i < width * height; ++i)
    {
        // Channel G
        imageVector.push_back(static_cast<double>(image_data[1 + 3 * i]) / 255.0f);
    }
    for (int i = 0; i < width * height; ++i)
    {
        // Channel B
        imageVector.push_back(static_cast<double>(image_data[2 + 3 * i]) / 255.0f);
    }

    stbi_image_free(image_data);

    return imageVector;
}

std::vector<double> im2col(const std::vector<double> &input,
                           int batch_size, int channels, int height, int width,
                           int filter_h, int filter_w, int stride = 1, int padding = 0)
{
    int out_h = (height + 2 * padding - filter_h) / stride + 1;
    int out_w = (width + 2 * padding - filter_w) / stride + 1;

    std::vector<double> col(batch_size * out_h * out_w * channels * filter_h * filter_w, 0);

    for (int n = 0; n < batch_size; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int y = 0; y < filter_h; ++y)
            {
                for (int x = 0; x < filter_w; ++x)
                {
                    int col_row = 0;
                    for (int oy = 0; oy < out_h; ++oy)
                    {
                        int in_y = oy * stride + y - padding;
                        for (int ox = 0; ox < out_w; ++ox)
                        {
                            int in_x = ox * stride + x - padding;
                            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width)
                            {
                                int input_index = n * channels * height * width + c * height * width + in_y * width + in_x;
                                int col_index = (n * out_h * out_w + col_row) * (channels * filter_h * filter_w) + c * filter_h * filter_w + y * filter_w + x;
                                col[col_index] = input[input_index];
                            }
                            ++col_row;
                        }
                    }
                }
            }
        }
    }
    return col;
}

std::vector<double> im2col_transposed(const std::vector<double> &input,
                                      int batch_size, int channels, int height, int width,
                                      int filter_h, int filter_w, int stride = 1, int padding = 0)
{
    int out_h = (height + 2 * padding - filter_h) / stride + 1;
    int out_w = (width + 2 * padding - filter_w) / stride + 1;

    std::vector<double> col(batch_size * out_h * out_w * channels * filter_h * filter_w, 0);

    for (int n = 0; n < batch_size; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int y = 0; y < filter_h; ++y)
            {
                for (int x = 0; x < filter_w; ++x)
                {
                    int col_row = 0;
                    for (int oy = 0; oy < out_h; ++oy)
                    {
                        int in_y = oy * stride + y - padding;
                        for (int ox = 0; ox < out_w; ++ox)
                        {
                            int in_x = ox * stride + x - padding;
                            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width)
                            {
                                int input_index = n * channels * height * width + c * height * width + in_y * width + in_x;
                                int col_index = (c * filter_h * filter_w + y * filter_w + x) * (batch_size * out_h * out_w) + n * out_h * out_w + col_row;
                                col[col_index] = input[input_index];
                            }
                            ++col_row;
                        }
                    }
                }
            }
        }
    }
    return col;
}


std::vector<double> kernel2col(const std::vector<double> &kernel,
                               int out_channels, int in_channels,
                               int filter_h, int filter_w)
{
    std::vector<double> col(in_channels * filter_h * filter_w * out_channels);

    for (int oc = 0; oc < out_channels; ++oc)
    {
        int col_index = 0;
        for (int ic = 0; ic < in_channels; ++ic)
        {
            for (int y = 0; y < filter_h; ++y)
            {
                for (int x = 0; x < filter_w; ++x)
                {
                    int kernel_index = oc * in_channels * filter_h * filter_w + ic * filter_h * filter_w + y * filter_w + x;
                    int col_flat_index = col_index * out_channels + oc; 
                    col[col_flat_index] = kernel[kernel_index];
                    ++col_index;
                }
            }
        }
    }

    return col;
}

std::vector<double> kernel2col_transposed(const std::vector<double> &kernel,
                                          int out_channels, int in_channels,
                                          int filter_h, int filter_w)
{
    std::vector<double> col(in_channels * filter_h * filter_w * out_channels);

    for (int oc = 0; oc < out_channels; ++oc)
    {
        int col_index = 0;
        for (int ic = 0; ic < in_channels; ++ic)
        {
            for (int y = 0; y < filter_h; ++y)
            {
                for (int x = 0; x < filter_w; ++x)
                {
                    int kernel_index = oc * in_channels * filter_h * filter_w + ic * filter_h * filter_w + y * filter_w + x;
                    int col_flat_index = oc * (in_channels * filter_h * filter_w) + col_index;
                    col[col_flat_index] = kernel[kernel_index];
                    ++col_index;
                }
            }
        }
    }

    return col;
}

std::vector<double> matrix_multiply(const std::vector<double> &A,
                                    const std::vector<double> &B, int m, int p, int n)
{
    int rows = m;
    int cols = n;
    int inner_dim = p;

    std::vector<double> result(rows * cols, 0);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            for (int k = 0; k < inner_dim; ++k)
            {
                result[i * cols + j] += A[i * inner_dim + k] * B[k * cols + j];
            }
        }
    }
    return result;
}

int main()
{

    vector<double> input = read_image("../airplane.png");
    vector<double> kernel = read_file("../weights/model_init_conv1_fused.txt");
    vector<double> bias = read_file("../weights/model_init_bn1_fused.txt");

    int batch_size = 1, in_channels = 3, height = 32, width = 32;
    int filter_h = 3, filter_w = 3, stride = 1, padding = 1;
    int out_channels = 16, out_h = (height + 2 * padding - filter_h) / stride + 1, out_w = (width + 2 * padding - filter_w) / stride + 1;

    // int batch_size = 1;
    // int channels = 1;
    // int height = 4;
    // int width = 4;
    // int filter_h = 2;
    // int filter_w = 2;
    // int stride = 1;
    // int padding = 0;

    auto col = im2col(input, batch_size, in_channels, height, width, filter_h, filter_w, stride, padding);
    auto col_transposed = im2col_transposed(input, batch_size, in_channels, height, width, filter_h, filter_w, stride, padding);
    auto col_kernel = kernel2col(kernel, out_channels, in_channels, filter_h, filter_w);
    auto col_kernel_transposed = kernel2col_transposed(kernel, out_channels, in_channels, filter_h, filter_w);

    int col_width = in_channels * filter_h * filter_w;

    for (auto &val : col)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;


    std::cout << "========== input ================" << std::endl;
    
    for (int i = 0; i < batch_size * out_h * out_w; ++i)
    {
        for (int j = 0; j < col_width; ++j)
        {
            std::cout << col[i * col_width + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "========== input reverse ================" << std::endl;
    for (int i = 0; i < in_channels * filter_h * filter_w; ++i)
    {
        for (int j = 0; j < batch_size * out_h * out_w; ++j)
        {
            std::cout << col_transposed[i * batch_size * out_h * out_w + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "========== kernel ================" << std::endl;

    for (auto &val : col_kernel)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    int col_height = in_channels * filter_h * filter_w;
    col_width = out_channels;

    for (int i = 0; i < col_height; ++i)
    {
        for (int j = 0; j < col_width; ++j)
        {
            std::cout << col_kernel[i * col_width + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "========== kernel reverse ================" << std::endl;

    for (auto &val : col_kernel_transposed)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    col_height = out_channels;
    col_width = in_channels * filter_h * filter_w;

    printf("col_height = %d, col_width = %d\n", col_height, col_width);

    for (int i = 0; i < col_height; ++i)
    {
        for (int j = 0; j < col_width; ++j)
        {
            std::cout << col_kernel_transposed[i * col_width + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "========== matrix result ================" << std::endl;

    auto ret = matrix_multiply(col_kernel_transposed, col_transposed, out_channels, in_channels * filter_h * filter_w, batch_size * out_h * out_w);

    for (int i = 0; i < ret.size(); i++)
    {
        ret[i] += bias[i];
    }

    for (auto &val : ret)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
