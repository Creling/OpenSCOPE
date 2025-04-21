#include <vector>
#include <iostream>

double im2col_get_pixel(std::vector<double>& im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

double im2col_get_pixel_vec(double *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

// void im2col_cpu(double* data_im,
//      int channels,  int height,  int width,
//      int ksize,  int stride, int pad, double* data_col) 
// {
//     int c,h,w;
//     int height_col = (height + 2*pad - ksize) / stride + 1;
//     int width_col = (width + 2*pad - ksize) / stride + 1;

//     int channels_col = channels * ksize * ksize;
//     for (c = 0; c < channels_col; ++c) {
//         int w_offset = c % ksize;
//         int h_offset = (c / ksize) % ksize;
//         int c_im = c / ksize / ksize;
//         for (h = 0; h < height_col; ++h) {
//             for (w = 0; w < width_col; ++w) {
//                 int im_row = h_offset + h * stride;
//                 int im_col = w_offset + w * stride;
//                 int col_index = (c * height_col + h) * width_col + w;
//                 data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
//                         im_row, im_col, c_im, pad);
//             }
//         }
//     }
// }

#include <vector>

std::vector<double> im2col_cpu_vec(std::vector<double>& data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad) 
{
    int c, h, w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    std::vector<double> data_col;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
    return data_col;
}

std::vector<double> im2col(const std::vector<double>& input, 
                          int batch_size, int channels, int height, int width, 
                          int filter_h, int filter_w, int stride = 1, int padding = 0) {
    int out_h = (height + 2 * padding - filter_h) / stride + 1;
    int out_w = (width + 2 * padding - filter_w) / stride + 1;
    
    std::vector<double> col(batch_size * out_h * out_w * channels * filter_h * filter_w, 0);

    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < filter_h; ++y) {
                for (int x = 0; x < filter_w; ++x) {
                    int col_row = 0;
                    for (int oy = 0; oy < out_h; ++oy) {
                        int in_y = oy * stride + y - padding;
                        for (int ox = 0; ox < out_w; ++ox) {
                            int in_x = ox * stride + x - padding;
                            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
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

int main() {
    std::vector<double> input = {1, 2, 3, 4,
                                5, 6, 7, 8,
                                9, 10, 11, 12,
                                13, 14, 15, 16};
    int batch_size = 1;
    int channels = 1;
    int height = 4;
    int width = 4;
    int filter_h = 2;
    int filter_w = 2;
    int stride = 1;
    int padding = 1;
    
    auto col = im2col(input, batch_size, channels, height, width, filter_h, filter_w, stride, padding);
// 
    // auto col = im2col_cpu_vec(input, channels, height, width, filter_h, stride, padding);


    int out_h = (height + 2 * padding - filter_h) / stride + 1;
    int out_w = (width + 2 * padding - filter_w) / stride + 1;
    int col_width = channels * filter_h * filter_w;

    for (auto &val : col) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < batch_size * out_h * out_w; ++i) {
        for (int j = 0; j < col_width; ++j) {
            std::cout << col[i * col_width + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
