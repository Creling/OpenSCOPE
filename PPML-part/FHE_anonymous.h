#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <optional>
#include <random>
#include <iostream>
#include <vector>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

// #include "test_adv.h"
// #include "test.h"
#include "troy/troy.h"

#include "utils.h"

#include <fstream>
#include <tuple>
#include <algorithm>
#include <numeric>

using namespace troy;
using namespace troy::linear;
using std::stringstream;
using std::vector;
using namespace std;

std::vector<double> calculate_averages(const std::vector<double> &data, size_t group_size)
{
    std::vector<double> averages;
    size_t num_groups = data.size() / group_size;

    for (size_t i = 0; i < num_groups; ++i)
    {
        double sum = std::accumulate(data.begin() + i * group_size, data.begin() + (i + 1) * group_size, 0.0);
        averages.push_back(sum / group_size);
    }

    return averages;
}

void writeToFile(const std::vector<double> &data, const std::string &filename)
{
    std::ofstream outFile(filename);
    if (!outFile)
    {
        return;
    }

    for (const auto &value : data)
    {
        outFile << value << ",";
    }

    outFile.close();
}

class ResnetContext
{
private:
    EncryptionParameters &parms;
    HeContextPointer &context;
    Encryptor &encryptor;
    Evaluator &evaluator;
    Decryptor &decryptor;
    CKKSEncoder &encoder;
    PublicKey &public_key;
    SecretKey &secret_key;
    RelinKeys &relin_keys;
    GaloisKeys &gal_keys;
    double scale;

public:
    Role role;
    int socket_fd;
    vector<double> cache;
    vector<double> selected_random_vector_cache;
    vector<double> fake_weight;
    vector<double> fake_bias;

    ResnetContext(EncryptionParameters &parms, HeContextPointer &context, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, GaloisKeys &gal_keys)
        : parms(parms), context(context), encryptor(encryptor), evaluator(evaluator), decryptor(decryptor), encoder(encoder), public_key(public_key), secret_key(secret_key), relin_keys(relin_keys), gal_keys(gal_keys)
    {
        scale = (1 << 20);
        role = Role::Server;
        cache = vector<double>(65535);
        selected_random_vector_cache = vector<double>(65535);
        fake_weight = vector<double>(6553500, 1.0);
        fake_bias = vector<double>(6553500, 1.0);
    }

    vector<double> relu_anonymous_simple(Cipher2d &in, MatmulHelper &decrypt_helper)
    {
        vector<double> y_decrypted = decrypt_helper.decrypt_outputs_doubles(encoder, decryptor, in);
        std::cout << "y_decrypted size: " << y_decrypted.size() << std::endl;
        for (size_t i = 0; i < y_decrypted.size(); i++)
        {
            y_decrypted[i] = std::max(0.0, y_decrypted[i]);
        }
        writeToFile(y_decrypted, "relu.txt");

        return y_decrypted;
    }

    Cipher2d Conv2d(string weight_path, string bias_path, Cipher2d &in, Conv2dHelper &helper)
    {
        vector<double> weight = read_file(weight_path);
        vector<double> bias = read_file(bias_path);

        Plain2d weight_encoded = helper.encode_weights_doubles(encoder, weight.data(), std::nullopt, scale);
        Plain2d bias_encoded = helper.encode_outputs_doubles(encoder, bias.data(), std::nullopt, scale * scale);

        Cipher2d y_encrypted = helper.conv2d(evaluator, in, weight_encoded);
        y_encrypted.add_plain_inplace(evaluator, bias_encoded);

        return y_encrypted;
    }

    Cipher2d Conv2d_fake(string weight_path, string bias_path, Cipher2d &in, Conv2dHelper &helper)
    {
        printf("enter Conv2d_fake\n");

        auto start_time = std::chrono::high_resolution_clock::now();

        Plain2d weight_encoded = helper.encode_weights_doubles(encoder, fake_weight.data(), std::nullopt, scale);
        // std::cout << "weight_encoded: " << weight_encoded.data().size() << std::endl;

        Plain2d bias_encoded = helper.encode_outputs_doubles(encoder, fake_bias.data(), std::nullopt, scale * scale);
        // std::cout << "bias_encoded: " << bias_encoded.data().size() << std::endl;

        Cipher2d y_encrypted = helper.conv2d(evaluator, in, weight_encoded);
        y_encrypted.add_plain_inplace(evaluator, bias_encoded);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        printf("Conv2d_fake elapsed time: %f\n", elapsed.count());

        return y_encrypted;
    }

    Cipher2d relu_anonymous_interact_add(Cipher2d &in, CKKSEncoder &encoder, Decryptor &decryptor, Conv2dHelper &curHelper, Conv2dHelper &nextHelper, int nextH, int nextW, int nextC, int nextPadding, bool cache = false)
    {
        printf("Enter relu_anonymous_interact_add\n");
        if (role == Role::Server)
        {
            printf("Enter relu_anonymous_interact_add S\n");

            auto random_mask = get_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            auto random_mask_encoded = curHelper.encode_outputs_doubles(encoder, random_mask.data(), std::nullopt, scale);

            in.add_plain_inplace(evaluator, random_mask_encoded);

            if (cache)
            {
                this->selected_random_vector_cache = get_selected_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            }
            
            stringstream ct_stream;
            in.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);
            
            ct_stream.str("");
            ct_stream.clear();
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);

            selected_random_mask = get_selected_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            selected_random_mask_encoded = curHelper.encode_weights_doubles(encoder, selected_random_mask.data(), std::nullopt, scale);
            selected_random_mask_encoded_2 = curHelper.encode_weights_doubles(encoder, this->selected_random_vector_cache.data(), std::nullopt, scale);
            
            deserialized.sub_plain_inplace(evaluator, selected_random_mask);
            deserialized.sub_plain_inplace(evaluator, selected_random_mask_encoded_2);

            return deserialized;
        }
        else
        {
            printf("Enter relu_anonymous_interact_add C\n");

            stringstream ct_stream;
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);
            printf("deserialized.load(ct_stream, context);\n");
            
            vector<double> y_decrypted = curHelper.decrypt_outputs_doubles(encoder, decryptor, deserialized);
            for (size_t i = 0; i < y_decrypted.size(); i++)
            {
                y_decrypted[i] += this->cache[i];
                y_decrypted[i] = std::max(0.0, y_decrypted[i]);                                                  
            }
            writeToFile(y_decrypted, "relu_anonymous_interact_add.txt");

            if (cache)
            {
                this->cache = y_decrypted;
            }

            vector<double> padded_image = pad_image(y_decrypted, nextW, nextH, nextC, nextPadding);
            Plain2d image_encoded = nextHelper.encode_inputs_doubles(encoder, padded_image.data(), std::nullopt, scale);
            Cipher2d image_encrypted = image_encoded.encrypt_asymmetric(encryptor);
            
            ct_stream.str("");
            ct_stream.clear();
            image_encrypted.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);

            return deserialized;
        }
    }

    Cipher2d relu_anonymous_interact_add_avgpool(Cipher2d &in, CKKSEncoder &encoder, Decryptor &decryptor, Conv2dHelper &curHelper, MatmulHelper &nextHelper, int nextH, int nextW, int nextC, int nextPadding, bool cache = false)
    {
        printf("Enter relu_anonymous_interact_add\n");
        if (role == Role::Server)
        {
            printf("Enter relu_anonymous_interact_add S\n");
            
            stringstream ct_stream;

            random_mask = get_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            random_mask_encoded = curHelper.encode_outputs_doubles(encoder, random_mask.data(), std::nullopt, scale);
            in.add_plain_inplace(evaluator, random_mask_encoded);

            in.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);
            
            ct_stream.str("");
            ct_stream.clear();
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);

            selected_random_mask = get_selected_random_vector_avgpool(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            selected_random_mask_encoded = curHelper.encode_weights_doubles(encoder, selected_random_mask.data(), std::nullopt, scale);
            deserialized.sub_plain_inplace(evaluator, selected_random_mask);

            return deserialized;
        }
        else
        {
            printf("Enter relu_anonymous_interact_add C\n");

            stringstream ct_stream;
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);
            printf("deserialized.load(ct_stream, context);\n");
            
            vector<double> y_decrypted = curHelper.decrypt_outputs_doubles(encoder, decryptor, deserialized);
            for (size_t i = 0; i < y_decrypted.size(); i++)
            {
                y_decrypted[i] += this->cache[i];
                y_decrypted[i] = std::max(0.0, y_decrypted[i]);                                                  
            }
            writeToFile(y_decrypted, "relu_anonymous_interact_add.txt");

            if (cache)
            {
                this->cache = y_decrypted;
            }

            // avgpool

            printf("y_decrypted.size(): %lu\n", y_decrypted.size());
            std::vector<double> y_decrypted_avg = calculate_averages(y_decrypted, 64);
            printf("y_decrypted_avg.size(): %lu\n", y_decrypted_avg.size());
            writeToFile(y_decrypted_avg, "relu_anonymous_interact_add_avgpool.txt");

            // end avgpool


            Plain2d image_encoded = nextHelper.encode_weights_doubles(encoder, y_decrypted_avg.data(), std::nullopt, scale);
            Cipher2d image_encrypted = image_encoded.encrypt_asymmetric(encryptor);
            
            ct_stream.str("");
            ct_stream.clear();
            image_encrypted.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);

            return deserialized;
        }
    }

    Cipher2d relu_anonymous_interact_add_double(Cipher2d &in, Cipher2d &out, CKKSEncoder &encoder, Decryptor &decryptor, Conv2dHelper &curHelper, Conv2dHelper &nextHelper_1, Conv2dHelper &nextHelper_2, int nextH, int nextW, int nextC, int nextPadding, bool cache = false)
    {
        printf("Enter relu_anonymous_interact_add\n");
        if (role == Role::Server)
        {
            printf("Enter relu_anonymous_interact_add S\n");
            
            stringstream ct_stream;

            auto random_mask = get_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            auto random_mask_encoded = curHelper.encode_outputs_doubles(encoder, random_mask.data(), std::nullopt, scale);

            in.add_plain_inplace(evaluator, random_mask_encoded);

            in.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);

            if (cache)
            {
                this->selected_random_vector_cache = get_selected_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            }

            ct_stream.str("");
            ct_stream.clear();
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);

            ct_stream.str("");
            ct_stream.clear();
            ct_stream = receiveStream(socket_fd);
            out.load(ct_stream, context);

            auto selected_random_mask = get_selected_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            auto selected_random_mask_encoded = curHelper.encode_inputs_doubles(encoder, selected_random_mask.data(), std::nullopt, scale);
            deserialized.sub_plain_inplace(evaluator, selected_random_mask);
            deserialized.sub_plain_inplace(evaluator, selected_random_mask_encoded_2);

            return deserialized;
        }
        else
        {
            printf("Enter relu_anonymous_interact_add C\n");

            stringstream ct_stream;
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);
            printf("deserialized.load(ct_stream, context);\n");
            
            vector<double> y_decrypted = curHelper.decrypt_outputs_doubles(encoder, decryptor, deserialized);
            for (size_t i = 0; i < y_decrypted.size(); i++)
            {
                y_decrypted[i] += this->cache[i];
                y_decrypted[i] = std::max(0.0, y_decrypted[i]);                                                  
            }
            writeToFile(y_decrypted, "relu_anonymous_interact_add_double.txt");

            if (cache)
            {
                this->cache = y_decrypted;
            }

            vector<double> padded_image = pad_image(y_decrypted, nextW, nextH, nextC, nextPadding);
            Plain2d image_encoded = nextHelper_1.encode_inputs_doubles(encoder, padded_image.data(), std::nullopt, scale);
            Cipher2d image_encrypted = image_encoded.encrypt_asymmetric(encryptor);
            
            ct_stream.str("");
            ct_stream.clear();
            image_encrypted.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);


            padded_image = pad_image(y_decrypted, nextW, nextH, nextC, 0);
            writeToFile(padded_image, "relu_anonymous_interact_add_1_pad_image.txt");
            image_encoded = nextHelper_2.encode_inputs_doubles(encoder, padded_image.data(), std::nullopt, scale);
            image_encrypted = image_encoded.encrypt_asymmetric(encryptor);

            // //
            // auto ret2 = Conv2d("../weights/model_layer2_block0_convbn3.txt", "../weights/model_layer2_block0_bias3.txt", image_encrypted, help_16_32_1_2_0); // 32768
            // auto ret2_decrypted = help_16_32_1_2_0.decrypt_outputs_doubles(encoder, decryptor, ret2);
            // writeToFile(ret2_decrypted, "conv2d_anonymous_im2col_layer2_block0_convbn3_client.txt");
            // //
                                                                                        
            ct_stream.str("");
            ct_stream.clear();
            image_encrypted.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);

            return deserialized;
        }
    }

    Cipher2d relu_anonymous_interact_downsample_add(Cipher2d &in, Cipher2d &in2, CKKSEncoder &encoder, Decryptor &decryptor, Conv2dHelper &curHelper_1, Conv2dHelper &curHelper_2, Conv2dHelper &nextHelper, int nextH, int nextW, int nextC, int nextPadding, bool cache = false)
    {
        printf("Enter relu_anonymous_interact_add\n");
        if (role == Role::Server)
        {
            printf("Enter relu_anonymous_interact_add S\n");
            
            stringstream ct_stream;

            auto random_mask = get_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            auto random_mask_encoded = curHelper.encode_outputs_doubles(encoder, random_mask.data(), std::nullopt, scale);

            in.add_plain_inplace(evaluator, random_mask_encoded);

            in.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);

            if (cache)
            {
                this->selected_random_vector_cache = get_selected_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            }

            ct_stream.str("");
            ct_stream.clear();
            in2.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);

            ct_stream.str("");
            ct_stream.clear();
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);

            auto selected_random_mask = get_selected_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            auto selected_random_mask_encoded = curHelper.encode_inputs_doubles(encoder, selected_random_mask.data(), std::nullopt, scale);
            deserialized.sub_plain_inplace(evaluator, selected_random_mask);
            deserialized.sub_plain_inplace(evaluator, selected_random_mask_encoded_2);

            return deserialized;
        }
        else
        {
            printf("Enter relu_anonymous_interact_add C\n");

            stringstream ct_stream;
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context); // 8192
            printf("deserialized.load(ct_stream, context);\n");

            ct_stream.str("");
            ct_stream.clear();
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized_2;
            deserialized_2.load(ct_stream, context); // 32768
            printf("deserialized.load(ct_stream, context);\n");

            vector<double> y_decrypted = curHelper_1.decrypt_outputs_doubles(encoder, decryptor, deserialized);
            vector<double> y_decrypted_2 = curHelper_2.decrypt_outputs_doubles(encoder, decryptor, deserialized_2);

            vector<double> new_y_decrypted(nextC * nextW * nextH);
            for (int c = 0; c < nextC; ++c) {
                for (int i = 0; i < nextH; ++i) {
                    for (int j = 0; j < nextW; ++j) {
                        new_y_decrypted[c * nextH * nextW + i * nextW + j] = y_decrypted_2[c * (nextH * 2) * (nextW * 2) + 2 * i * (nextW * 2) + 2 * j];
                    }
                }
            }
            writeToFile(new_y_decrypted, "relu_anonymous_interact_downsample_add.txt");
            writeToFile(y_decrypted, "relu_anonymous_interact_downsample_add_y_decrypted.txt");
            
            for (size_t i = 0; i < y_decrypted.size(); i++)
            {
                y_decrypted[i] += new_y_decrypted[i];
                y_decrypted[i] = std::max(0.0, y_decrypted[i]);                                                  
            }
            writeToFile(y_decrypted, "relu_anonymous_interact_add.txt");

            if (cache)
            {
                this->cache = y_decrypted;
            }

            vector<double> padded_image = pad_image(y_decrypted, nextW, nextH, nextC, nextPadding);
            Plain2d image_encoded = nextHelper.encode_inputs_doubles(encoder, padded_image.data(), std::nullopt, scale);
            Cipher2d image_encrypted = image_encoded.encrypt_asymmetric(encryptor);

            ct_stream.str("");
            ct_stream.clear();
            image_encrypted.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);
            return deserialized;
        }
    }

    Cipher2d relu_anonymous_interact(Cipher2d &in, CKKSEncoder &encoder, Decryptor &decryptor, Conv2dHelper &curHelper, Conv2dHelper &nextHelper, int nextH, int nextW, int nextC, int nextChannel, bool cache = false)
    {
        if (role == Role::Server)
        {
            printf("Enter relu_anonymous_interact S\n");
            
            stringstream ct_stream;

            auto random_mask = get_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            auto random_mask_encoded = curHelper.encode_outputs_doubles(encoder, random_mask.data(), std::nullopt, scale);

            in.add_plain_inplace(evaluator, random_mask_encoded);

            in.save(ct_stream, context, CompressionMode::Zstd);
            // in.save(ct_stream, context);
            sendStream(socket_fd, ct_stream);
            
            ct_stream.str("");
            ct_stream.clear();
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);

            auto selected_random_mask = get_selected_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            auto selected_random_mask_encoded = curHelper.encode_inputs_doubles(encoder, selected_random_mask.data(), std::nullopt, scale);
            deserialized.sub_plain_inplace(evaluator, selected_random_mask);

            return deserialized;
        }
        else
        {
            printf("Enter relu_anonymous_interact C\n");

            stringstream ct_stream;
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);

            vector<double> y_decrypted = curHelper.decrypt_outputs_doubles(encoder, decryptor, deserialized);
            for (size_t i = 0; i < y_decrypted.size(); i++)
            {
                y_decrypted[i] = std::max(0.0, y_decrypted[i]);
            }
            // writeToFile(y_decrypted, "relu_anonymous_interact_client.txt");

            if (cache)
            {
                this->cache = y_decrypted;
            }   

            y_decrypted = vector<double>(6553500);
            vector<double> padded_image = pad_image(y_decrypted, nextW, nextH, nextC, nextChannel);
            // std::cout << "padded_image size: " << padded_image.size() << std::endl;
            Plain2d image_encoded = nextHelper.encode_inputs_doubles(encoder, y_decrypted.data(), std::nullopt, scale);
            Cipher2d image_encrypted = image_encoded.encrypt_asymmetric(encryptor);
            
            ct_stream.str("");
            ct_stream.clear();
            image_encrypted.save(ct_stream, context, CompressionMode::Zstd);
            
            sendStream(socket_fd, ct_stream);

            return deserialized;
        }
    }

    Cipher2d relu_anonymous_interact_downsample(Cipher2d &in, CKKSEncoder &encoder, Decryptor &decryptor, Conv2dHelper &curHelper, Conv2dHelper &nextHelper, int nextH, int nextW, int nextC, int nextPadding, bool cache = false)
    {
        printf("Enter relu_anonymous_interact\n");
        if (role == Role::Server)
        {
            printf("Enter relu_anonymous_interact S\n");
            
            stringstream ct_stream;

            auto random_mask = get_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            auto random_mask_encoded = curHelper.encode_outputs_doubles(encoder, random_mask.data(), std::nullopt, scale);

            in.add_plain_inplace(evaluator, random_mask_encoded);

            if (cache)
            {
                this->selected_random_vector_cache = get_selected_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            }

            in.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);
            
            ct_stream.str("");
            ct_stream.clear();
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);

            auto selected_random_mask = get_selected_random_vector(curHelper, nextHelper, nextH, nextW, nextC, nextPadding);
            auto selected_random_mask_encoded = curHelper.encode_inputs_doubles(encoder, selected_random_mask.data(), std::nullopt, scale);
            deserialized.sub_plain_inplace(evaluator, selected_random_mask);

            return deserialized;
        }
        else
        {
            printf("Enter relu_anonymous_interact C\n");

            stringstream ct_stream;
            ct_stream = receiveStream(socket_fd);
            Cipher2d deserialized;
            deserialized.load(ct_stream, context);
            printf("deserialized.load(ct_stream, context);\n");

            PRINT_2D_VECTOR_SHAPE(deserialized.data());

            vector<double> y_decrypted = curHelper.decrypt_outputs_doubles(encoder, decryptor, deserialized);
            writeToFile(y_decrypted, "relu_anonymous_interact_downsample_1.txt");

            vector<double> new_y_decrypted(nextC * nextW * nextH);
            for (int c = 0; c < nextC; ++c) {
                for (int i = 0; i < nextH; ++i) {
                    for (int j = 0; j < nextW; ++j) {
                        new_y_decrypted[c * nextH * nextW + i * nextW + j] = y_decrypted[c * (nextH * 2) * (nextW * 2) + 2 * i * (nextW * 2) + 2 * j];
                    }
                }
            }

            writeToFile(new_y_decrypted, "relu_anonymous_interact_downsample_2.txt");

            for (size_t i = 0; i < new_y_decrypted.size(); i++)
            {
                new_y_decrypted[i] = std::max(0.0, new_y_decrypted[i]);
            }

            writeToFile(new_y_decrypted, "relu_anonymous_interact_downsample_3.txt");

            if (cache)
            {
                this->cache = y_decrypted;
            }   

            vector<double> padded_image = pad_image(new_y_decrypted, nextW, nextH, nextC, nextPadding);
            Plain2d image_encoded = nextHelper.encode_inputs_doubles(encoder, padded_image.data(), std::nullopt, scale);
            Cipher2d image_encrypted = image_encoded.encrypt_asymmetric(encryptor);
            
            ct_stream.str("");
            ct_stream.clear();
            image_encrypted.save(ct_stream, context, CompressionMode::Zstd);
            sendStream(socket_fd, ct_stream);

            return deserialized;
        }
    }

    Cipher2d relu_anonymous_interact_no(Cipher2d &in, CKKSEncoder &encoder, Decryptor &decryptor, Conv2dHelper &curHelper, Conv2dHelper &nextHelper, int nextH, int nextW, int nextC, int nextPadding)
    {
        stringstream ct_stream;
        in.save(ct_stream, context, CompressionMode::Zstd);
        Cipher2d deserialized;
        deserialized.load(ct_stream, context);

        vector<double> y_decrypted = curHelper.decrypt_outputs_doubles(encoder, decryptor, in);
        // vector<double> y_decrypted = curHelper.decrypt_outputs_doubles(encoder, decryptor, in);

        for (size_t i = 0; i < y_decrypted.size(); i++)
        {
            y_decrypted[i] = std::max(0.0, y_decrypted[i]);
        }
        writeToFile(y_decrypted, "relu_anonymous_interact_no.txt");

        vector<double> padded_image = pad_image(y_decrypted, nextW, nextH, nextC, nextPadding);
        Plain2d image_encoded = nextHelper.encode_inputs_doubles(encoder, padded_image.data(), std::nullopt, scale);
        Cipher2d image_encrypted = image_encoded.encrypt_asymmetric(encryptor);

        return image_encrypted;
    }

    Cipher2d relu_anonymous_noreshape(Cipher2d &in, MatmulHelper &decrypt_helper, MatmulHelper &encrype_helper, int batch_size, int in_channels, int out_channels, int height, int width,
                                  int filter_h, int filter_w, int stride, int padding)
    {
        vector<double> y_decrypted = decrypt_helper.decrypt_outputs_doubles(encoder, decryptor, in);
        std::cout << "y_decrypted size: " << y_decrypted.size() << std::endl;

        for (size_t i = 0; i < y_decrypted.size(); i++)
        {
            y_decrypted[i] = std::max(0.0, y_decrypted[i]);
        }
        writeToFile(y_decrypted, "relu_input.txt");

        Plain2d in_encoded = encrype_helper.encode_outputs_doubles(encoder, y_decrypted.data(), std::nullopt, scale * scale);
        Cipher2d in_encrypted = in_encoded.encrypt_asymmetric(encryptor);

        return in_encrypted;
    }

    Cipher2d convbn_initial_im2col(vector<double> &in, int batch_size, int in_channels, int out_channels, int height, int width,
                                   int filter_h, int filter_w, int stride = 1, int padding = 0)
    {
        int out_h = (height + 2 * padding - filter_h) / stride + 1;
        int out_w = (width + 2 * padding - filter_w) / stride + 1;
        int row = out_channels;
        int col = batch_size * out_h * out_w;
        int inner_dim = in_channels * filter_h * filter_w;

        // kernel_im2col [out_channels][in_channels * filter_h * filter_w]
        // input_im2col [in_channels * filter_h * filter_w][batch_size * out_h * out_w]

        MatmulHelper helper(row, inner_dim, col, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        std::cout << helper << endl;

        vector<double> weight = read_file("../weights/model_init_conv1_fused.txt");
        vector<double> bias = read_file("../weights/model_init_bn1_fused.txt");
        vector<double> kernel_im2col = kernel2col_transposed(weight, out_channels, in_channels, filter_h, filter_w);

        Plain2d kernel_im2col_encoded = helper.encode_inputs_doubles(encoder, kernel_im2col.data(), std::nullopt, scale);
        Plain2d input_im2col_encoded = helper.encode_weights_doubles(encoder, in.data(), std::nullopt, scale);
        Plain2d bias_encoded = helper.encode_outputs_doubles(encoder, bias.data(), std::nullopt, scale * scale);

        Cipher2d input_im2col_encrypted = input_im2col_encoded.encrypt_asymmetric(encryptor);

        Cipher2d y_encrypted = helper.matmul_reverse(evaluator, kernel_im2col_encoded, input_im2col_encrypted);
        y_encrypted.add_plain_inplace(evaluator, bias_encoded);

        // std::cout << "========== kernel reverse ================" << std::endl;

        // for (auto &val : kernel_im2col)
        // {
        //     std::cout << val << " ";
        // }
        // std::cout << std::endl;

        // int col_height = out_channels;
        // int col_width = in_channels * filter_h * filter_w;

        // printf("col_height=%d, col_width=%d\n", col_height, col_width);

        // std::cout << "========== conv results ================" << std::endl;

        // auto y_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, y_encrypted);

        // for (auto &var : y_decrypted)
        // {
        //     cout << var << " ";
        // }
        // cout << endl;

        return y_encrypted;
    }

    Cipher2d convbn2d(vector<double> &in, int batch_size, int in_channels, int out_channels, int height, int width,
                      int filter_h, int filter_w, int stride, int padding, vector<double> &weight, vector<double> &bias, MatmulHelper &helper)
    {
        int out_h = (height + 2 * padding - filter_h) / stride + 1;
        int out_w = (width + 2 * padding - filter_w) / stride + 1;
        int row = out_channels;
        int col = batch_size * out_h * out_w;
        int inner_dim = in_channels * filter_h * filter_w;

        // kernel_im2col [out_channels][in_channels * filter_h * filter_w]
        // input_im2col [in_channels * filter_h * filter_w][batch_size * out_h * out_w]

        helper = MatmulHelper(row, inner_dim, col, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        std::cout << helper << endl;

        // vector<double> weight = read_file("../weights/model_init_conv1_fused.txt");
        // vector<double> bias = read_file("../weights/model_init_bn1_fused.txt");
        vector<double> kernel_im2col = kernel2col_transposed(weight, out_channels, in_channels, filter_h, filter_w);

        Plain2d kernel_im2col_encoded = helper.encode_inputs_doubles(encoder, kernel_im2col.data(), std::nullopt, scale);
        Plain2d input_im2col_encoded = helper.encode_weights_doubles(encoder, in.data(), std::nullopt, scale);
        Plain2d bias_encoded = helper.encode_outputs_doubles(encoder, bias.data(), std::nullopt, scale * scale);

        Cipher2d input_im2col_encrypted = input_im2col_encoded.encrypt_asymmetric(encryptor);

        Cipher2d y_encrypted = helper.matmul_reverse(evaluator, kernel_im2col_encoded, input_im2col_encrypted);

        // auto y_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, y_encrypted);
        // for (auto &var : y_decrypted)
        // {
        //     cout << var << " ";
        // }
        // cout << endl;

        y_encrypted.add_plain_inplace(evaluator, bias_encoded);

        return y_encrypted;
    }

    Cipher2d convbn2d_v2(Cipher2d &in, int batch_size, int in_channels, int out_channels, int height, int width,
                         int filter_h, int filter_w, int stride, int padding, vector<double> &weight, vector<double> &bias, MatmulHelper &helper)
    {
        int out_h = (height + 2 * padding - filter_h) / stride + 1;
        int out_w = (width + 2 * padding - filter_w) / stride + 1;
        int row = out_channels;
        int col = batch_size * out_h * out_w;
        int inner_dim = in_channels * filter_h * filter_w;

        // kernel_im2col [out_channels][in_channels * filter_h * filter_w]
        // input_im2col [in_channels * filter_h * filter_w][batch_size * out_h * out_w]

        helper = MatmulHelper(row, inner_dim, col, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        std::cout << __LINE__ << helper << endl;

        // vector<double> weight = read_file("../weights/model_init_conv1_fused.txt");
        // vector<double> bias = read_file("../weights/model_init_bn1_fused.txt");
        vector<double> kernel_im2col = kernel2col_transposed(weight, out_channels, in_channels, filter_h, filter_w);
        std::cout << __LINE__ << helper << endl;
        Plain2d kernel_im2col_encoded = helper.encode_inputs_doubles(encoder, kernel_im2col.data(), std::nullopt, scale);
        std::cout << __LINE__ << helper << endl;
        Plain2d bias_encoded = helper.encode_outputs_doubles(encoder, bias.data(), std::nullopt, scale * scale);
        std::cout << __LINE__ << helper << endl;
        Cipher2d y_encrypted = helper.matmul_reverse(evaluator, kernel_im2col_encoded, in);
        std::cout << __LINE__ << helper << endl;
        // auto y_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, y_encrypted);
        // for (auto &var : y_decrypted)
        // {
        //     cout << var << " ";
        // }
        // cout << endl;

        y_encrypted.add_plain_inplace(evaluator, bias_encoded);

        return y_encrypted;
    }

    Cipher2d reshape(Cipher2d &in, MatmulHelper &decrypt_helper, MatmulHelper &encrype_helper, int batch_size, int in_channels, int out_channels, int height, int width,
                     int filter_h, int filter_w, int stride, int padding, bool relu = false)
    {
        vector<double> in_plain = decrypt_helper.decrypt_outputs_doubles(encoder, decryptor, in);
        if (relu)
        {
            for (size_t i = 0; i < in_plain.size(); i++)
            {
                in_plain[i] = std::max(0.0, in_plain[i]);
            }
            writeToFile(in_plain, "reshape_relu.txt");
        }

        auto in_im2col = im2col_transposed(in_plain, batch_size, in_channels, height, width, filter_h, filter_w, stride, padding);
        Plain2d in_encoded = encrype_helper.encode_weights_doubles(encoder, in_im2col.data(), std::nullopt, scale);
        Cipher2d in_encrypted = in_encoded.encrypt_asymmetric(encryptor);

        return in_encrypted;
    }

    void convbn(Cipher2d &in)
    {

        Conv2dHelper helper(1, 16, 16, 32 + 2, 32 + 2, 3, 3, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft);
        vector<double> weight = read_file("../weights/model_layer1_block0_conv1_fused.txt");
        Plain2d weight_encoded = helper.encode_weights_doubles(encoder, weight.data(), std::nullopt, scale); // std::nullopt means encode to level 0
        Cipher2d y_encrypted = helper.conv2d(evaluator, in, weight_encoded);
        vector<double> y_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, y_encrypted);
    }

    void convbn1(Cipher2d &in)
    {
    }

    ReturnValues layer2(Cipher2d &in, Cipher2d &in_noreshape)
    {
        std::cout << "========== layer 2 ================" << std::endl;

        std::cout << "=== block 0 ===" << std::endl;

        vector<double> weight = read_file("../weights/model_layer2_block0_convbn1.txt");
        vector<double> bias = read_file("../weights/model_layer2_block0_bias1.txt");

        MatmulHelper helper(1, 1, 1, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        auto in_back = in;

        auto ret = convbn2d_v2(in, 1, 16, 32, 32, 32, 3, 3, 2, 1, weight, bias, helper); // return shape: 32 * 256
        auto ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        MatmulHelper helper2(32, 288, 256, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
        ret = relu_anonymous(ret, helper, helper2, 1, 32, 32, 16, 16, 3, 3, 1, 1);

        weight = read_file("../weights/model_layer2_block0_convbn2.txt");
        bias = read_file("../weights/model_layer2_block0_bias2.txt");

        ret = convbn2d_v2(ret, 1, 32, 32, 16, 16, 3, 3, 1, 1, weight, bias, helper);
        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        // ...
        MatmulHelper decryptor_helper(16, 144, 1024, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
        MatmulHelper downsample_helper(32, 16, 256, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
        auto in_reshape = reshape(in_noreshape, decryptor_helper, downsample_helper, 1, 16, 32, 32, 32, 1, 1, 2, 0, false);
        printf("Line %d:\treshape Done\n", __LINE__);

        weight = read_file("../weights/model_layer2_block0_convbn3.txt");
        bias = read_file("../weights/model_layer2_block0_bias3.txt");
        Cipher2d in_downsampled = convbn2d_v2(in_reshape, 1, 16, 32, 32, 32, 1, 1, 2, 0, weight, bias, downsample_helper);
        auto in_downsampled_decrypted = downsample_helper.decrypt_outputs_doubles(encoder, decryptor, in_downsampled); // return shape: 32 * 256
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        MatmulHelper help_32_16_256 = MatmulHelper(32, 16, 256, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
        MatmulHelper help_32_288_256 = MatmulHelper(32, 288, 256, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        auto in_downsampled_reshape = help_32_288_256.encode_outputs_doubles(encoder, in_downsampled_decrypted.data(), std::nullopt, scale * scale);
        auto in_downsampled_encrypted = in_downsampled_reshape.encrypt_asymmetric(encryptor);

        // in_downsampled_decrypted = help_32_288_256.decrypt_outputs_doubles(encoder, decryptor, in_downsampled_encrypted); // return shape: 32 * 256

        ret.add_inplace(evaluator, in_downsampled_encrypted);
        std::cout << "add_inplace" << std::endl;

        // MatmulHelper help_32_288_1024 = MatmulHelper(32, 288, 256, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        Cipher2d ret_results = relu_anonymous(ret, help_32_288_256, help_32_288_256, 1, 32, 32, 16, 16, 3, 3, 1, 1);
        Cipher2d ret_noreshape = relu_anonymous_noreshape(ret, help_32_288_256, help_32_288_256, 1, 32, 32, 16, 16, 3, 3, 1, 1);
        vector<double> ret_noreshape_decrypted = help_32_288_256.decrypt_outputs_doubles(encoder, decryptor, ret_noreshape);
        writeToFile(ret_noreshape_decrypted, "conv2d_anonymous_im2col_layer2_block0_convbn3_relu.txt");

        std::cout << "=== block 1 ===" << std::endl;

        weight = read_file("../weights/model_layer2_block1_convbn1.txt");
        bias = read_file("../weights/model_layer2_block1_bias1.txt");

        ret = convbn2d_v2(ret_results, 1, 32, 32, 16, 16, 3, 3, 1, 1, weight, bias, helper); // return shape: 32 * 256
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);

        ret = relu_anonymous(ret, helper, helper, 1, 32, 32, 16, 16, 3, 3, 1, 1);
        printf("Line %d:\trelu_anonymous Done\n", __LINE__);

        weight = read_file("../weights/model_layer2_block1_convbn2.txt");
        bias = read_file("../weights/model_layer2_block1_bias2.txt");

        ret = convbn2d_v2(ret, 1, 32, 32, 16, 16, 3, 3, 1, 1, weight, bias, helper); // output shape: 32 * 256
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);

        ret.add_inplace(evaluator, ret_noreshape);

        ret_results = relu_anonymous(ret, help_32_288_256, help_32_288_256, 1, 32, 32, 16, 16, 3, 3, 1, 1);
        ret_noreshape = relu_anonymous_noreshape(ret, help_32_288_256, help_32_288_256, 1, 32, 32, 16, 16, 3, 3, 1, 1);
        ret_noreshape_decrypted = help_32_288_256.decrypt_outputs_doubles(encoder, decryptor, ret_noreshape);
        writeToFile(ret_noreshape_decrypted, "conv2d_anonymous_im2col_layer2_block1_convbn2_relu.txt");

        std::cout << "=== block 2 ===" << std::endl;

        weight = read_file("../weights/model_layer2_block2_convbn1.txt");
        bias = read_file("../weights/model_layer2_block2_bias1.txt");

        ret = convbn2d_v2(ret_results, 1, 32, 32, 16, 16, 3, 3, 1, 1, weight, bias, helper); // return shape: 32 * 256
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);

        ret = relu_anonymous(ret, helper, helper, 1, 32, 32, 16, 16, 3, 3, 1, 1);
        printf("Line %d:\trelu_anonymous Done\n", __LINE__);

        weight = read_file("../weights/model_layer2_block2_convbn2.txt");
        bias = read_file("../weights/model_layer2_block2_bias2.txt");

        ret = convbn2d_v2(ret, 1, 32, 32, 16, 16, 3, 3, 1, 1, weight, bias, helper); // output shape: 32 * 256
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);

        ret.add_inplace(evaluator, ret_noreshape);

        MatmulHelper help_64_288_64 = MatmulHelper(64, 288, 64, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        ret_results = relu_anonymous(ret, help_32_288_256, help_64_288_64, 1, 32, 64, 16, 16, 3, 3, 2, 1);
        ret_noreshape = relu_anonymous_noreshape(ret, help_32_288_256, help_32_288_256, 1, 32, 32, 16, 16, 3, 3, 1, 1);
        ret_noreshape_decrypted = help_32_288_256.decrypt_outputs_doubles(encoder, decryptor, ret_noreshape);

        return {ret_results, ret_noreshape};
    }

    ReturnValues layer3(Cipher2d &in, Cipher2d &in_noreshape)
    {
        std::cout << "========= layer 3 =========" << std::endl;

        std::cout << "====== block 0 ======" << std::endl;

        std::cout << "=== Convbn1 ===" << std::endl;

        vector<double> weight = read_file("../weights/model_layer3_block0_convbn1.txt");
        vector<double> bias = read_file("../weights/model_layer3_block0_bias1.txt");

        MatmulHelper helper(1, 1, 1, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        auto ret = convbn2d_v2(in, 1, 32, 64, 16, 16, 3, 3, 2, 1, weight, bias, helper); // return shape: 64 * 256
        auto ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        std::cout << "=== Relu ===" << std::endl;

        MatmulHelper helper2(64, 576, 64, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        std::cout << "=== Convbn2 ===" << std::endl;

        weight = read_file("../weights/model_layer3_block0_convbn2.txt");
        bias = read_file("../weights/model_layer3_block0_bias2.txt");

        ret = convbn2d_v2(ret, 1, 64, 64, 8, 8, 3, 3, 1, 1, weight, bias, helper); // output shape: 64 * 64
        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);   // return shape: 64 * 64
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        std::cout << "=== Downsample ===" << std::endl;

        weight = read_file("../weights/model_layer3_block0_convbn3.txt");
        bias = read_file("../weights/model_layer3_block0_bias3.txt");

        MatmulHelper decryptor_helper(32, 288, 256, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
        MatmulHelper downsample_helper(64, 32, 64, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
        printf("Line %d:\treshape Done\n", __LINE__);

        Cipher2d in_downsampled = convbn2d_v2(in_reshape, 1, 32, 64, 16, 16, 1, 1, 2, 0, weight, bias, downsample_helper);
        auto in_downsampled_decrypted = downsample_helper.decrypt_outputs_doubles(encoder, decryptor, in_downsampled); // return shape: 64 * 64
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        std::cout << "=== Add ===" << std::endl;

        std::cout << helper2 << endl;
        auto in_downsampled_reshape = helper2.encode_outputs_doubles(encoder, in_downsampled_decrypted.data(), std::nullopt, scale * scale);
        in_downsampled = in_downsampled_reshape.encrypt_asymmetric(encryptor);

        ret.add_inplace(evaluator, in_downsampled);

        std::cout << "=== Relu ===" << std::endl;

        MatmulHelper help_64_32_64 = MatmulHelper(64, 32, 64, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
        MatmulHelper help_64_576_64 = MatmulHelper(64, 576, 64, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);

        Cipher2d ret_results = relu_anonymous(ret, help_64_576_64, help_64_576_64, 1, 64, 64, 8, 8, 3, 3, 1, 1);
        Cipher2d ret_noreshape = relu_anonymous_noreshape(ret, help_64_576_64, help_64_576_64, 1, 64, 64, 8, 8, 3, 3, 1, 1);
        vector<double> ret_noreshape_decrypted = help_64_576_64.decrypt_outputs_doubles(encoder, decryptor, ret_noreshape);

        std::cout << "====== block 1 ======" << std::endl;

        std::cout << "=== Convbn1 ===" << std::endl;

        weight = read_file("../weights/model_layer3_block1_convbn1.txt");
        bias = read_file("../weights/model_layer3_block1_bias1.txt");

        ret = convbn2d_v2(ret_results, 1, 64, 64, 8, 8, 3, 3, 1, 1, weight, bias, helper); // return shape: 64 * 64
        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        std::cout << "=== Relu ===" << std::endl;

        ret = relu_anonymous(ret, helper, helper, 1, 64, 64, 8, 8, 3, 3, 1, 1);

        std::cout << "=== Convbn2 ===" << std::endl;

        weight = read_file("../weights/model_layer3_block1_convbn2.txt");
        bias = read_file("../weights/model_layer3_block1_bias2.txt");

        ret = convbn2d_v2(ret, 1, 64, 64, 8, 8, 3, 3, 1, 1, weight, bias, helper); // output shape: 64 * 64
        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);   // return shape: 64 * 64
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        std::cout << "=== Add ===" << std::endl;

        ret.add_inplace(evaluator, ret_noreshape);

        std::cout << "=== Relu ===" << std::endl;

        ret_results = relu_anonymous(ret, help_64_576_64, help_64_576_64, 1, 64, 64, 8, 8, 3, 3, 1, 1);
        ret_noreshape = relu_anonymous_noreshape(ret, help_64_576_64, help_64_576_64, 1, 64, 64, 8, 8, 3, 3, 1, 1);
        ret_noreshape_decrypted = help_64_576_64.decrypt_outputs_doubles(encoder, decryptor, ret_noreshape);

        std::cout << "====== block 2 ======" << std::endl;

        std::cout << "=== Convbn1 ===" << std::endl;

        weight = read_file("../weights/model_layer3_block2_convbn1.txt");
        bias = read_file("../weights/model_layer3_block2_bias1.txt");

        ret = convbn2d_v2(ret_results, 1, 64, 64, 8, 8, 3, 3, 1, 1, weight, bias, helper); // return shape: 64 * 64
        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        std::cout << "=== Relu ===" << std::endl;

        ret = relu_anonymous(ret, helper, helper, 1, 64, 64, 8, 8, 3, 3, 1, 1);

        std::cout << "=== Convbn2 ===" << std::endl;

        weight = read_file("../weights/model_layer3_block2_convbn2.txt");
        bias = read_file("../weights/model_layer3_block2_bias2.txt");

        ret = convbn2d_v2(ret, 1, 64, 64, 8, 8, 3, 3, 1, 1, weight, bias, helper); // output shape: 64 * 64
        ret_decrypted = helper.decrypt_outputs_doubles(encoder, decryptor, ret);   // return shape: 64 * 64
        printf("Line %d:\tconvbn2d_v2 Done\n", __LINE__);

        std::cout << "=== Add ===" << std::endl;

        ret.add_inplace(evaluator, ret_noreshape);

        std::cout << "=== Relu ===" << std::endl;

        ret_results = relu_anonymous(ret, help_64_576_64, help_64_576_64, 1, 64, 64, 8, 8, 3, 3, 1, 1);
        ret_noreshape = relu_anonymous_noreshape(ret, help_64_576_64, help_64_576_64, 1, 64, 64, 8, 8, 3, 3, 1, 1);
        ret_noreshape_decrypted = help_64_576_64.decrypt_outputs_doubles(encoder, decryptor, ret_noreshape);
        writeToFile(ret_noreshape_decrypted, "conv2d_anonymous_im2col_layer3_block2_convbn2_relu.txt"); //

        return {ret_results, ret_noreshape};
    }

    std::vector<double> calculate_averages(const std::vector<double> &data, size_t group_size)
    {
        std::vector<double> averages;
        size_t num_groups = data.size() / group_size;

        for (size_t i = 0; i < num_groups; ++i)
        {
            double sum = std::accumulate(data.begin() + i * group_size, data.begin() + (i + 1) * group_size, 0.0);
            averages.push_back(sum / group_size);
        }

        return averages;
    }

    vector<double> avgpool_layer(Cipher2d &in, Cipher2d &in_noreshape)
    {
        std::cout << "========= Avgpool =========" << std::endl;

        MatmulHelper help_64_576_64 = MatmulHelper(64, 576, 64, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
        // auto in_decrypted = help_64_576_64.decrypt_outputs_doubles(encoder, decryptor, in_noreshape);
        // writeToFile(in_decrypted, "conv2d_anonymous_im2col_avgpool.txt");

        vector<double> mask(parms.poly_modulus_degree(), 1.0);
        Plaintext mask_encoded = encoder.encode_float64_polynomial_new(mask, std::nullopt, scale * scale);

        vector<double> re_mask(parms.poly_modulus_degree(), 0);
        for (size_t i = 0; i < 64; i++)
        {
            re_mask[i] = 1.0;
        }
        Plaintext re_mask_encoded = encoder.encode_float64_polynomial_new(re_mask, std::nullopt, scale);

        auto in_noreshape_bak = in_noreshape;

        for (size_t i = 0; i < in_noreshape.size(); i++)
        {
            for (size_t j = 0; j < in_noreshape[i].size(); j++)
            {
                // auto temp_de = encoder.decode_float64_polynomial_new(decryptor.decrypt_new(in_noreshape[i][j]));
                // auto filename = "conv2d_anonymous_im2col_avgpool_" + std::to_string(i) + "_" + std::to_string(j) + ".txt";
                // writeToFile(temp_de, filename);
                evaluator.add_plain_inplace(in_noreshape[i][j], mask_encoded);
            }
        }

        // Client:

#ifdef DEBUG
        auto in_noreshape_bak_plain = encoder.decode_float64_polynomial_new(decryptor.decrypt_new(in_noreshape_bak[0][0]));
        writeToFile(in_noreshape_bak_plain, "conv2d_anonymous_im2col_avgpool_in_noreshape_bak.txt");

        auto in_noreshape_plain = encoder.decode_float64_polynomial_new(decryptor.decrypt_new(in_noreshape[0][0]));
        writeToFile(in_noreshape_plain, "conv2d_anonymous_im2col_avgpool_in_noreshape.txt");
#endif

        std::vector<double> masked_input = help_64_576_64.decrypt_outputs_doubles(encoder, decryptor, in_noreshape);
        writeToFile(masked_input, "conv2d_anonymous_im2col_avgpool2.txt");

#ifdef DEBUG
        std::vector<double> masked_input_bak = help_64_576_64.decrypt_outputs_doubles(encoder, decryptor, in_noreshape_bak);
        writeToFile(masked_input_bak, "conv2d_anonymous_im2col_avgpool2_bak.txt");
#endif

        std::vector<double> masked_avg = calculate_averages(masked_input, 64);

#ifdef DEBUG
        std::vector<double> masked_avg_bak = calculate_averages(masked_input_bak, 64);
        writeToFile(masked_avg_bak, "conv2d_anonymous_im2col_avgpool_averages_bak.txt");
#endif

        MatmulHelper help_10_64_1 = MatmulHelper(10, 64, 1, parms.poly_modulus_degree(), MatmulObjective::EncryptLeft, false);
        Plain2d masked_avg_encoded = help_10_64_1.encode_weights_doubles(encoder, masked_avg.data(), std::nullopt, scale);
        Cipher2d masked_avg_encrypted = masked_avg_encoded.encrypt_asymmetric(encryptor);

#ifdef DEBUG
        auto ori_input = help_64_576_64.decrypt_outputs_doubles(encoder, decryptor, in_noreshape_bak);
        std::vector<double> ori_avg = calculate_averages(ori_input, 64);
        Plain2d ori_avg_encoded = help_10_64_1.encode_weights_doubles(encoder, ori_avg.data(), std::nullopt, scale);
        Cipher2d ori_avg_encrypted = ori_avg_encoded.encrypt_asymmetric(encryptor);
#endif
        // Client Ends;

        Cipher2d avg_encrypted = masked_avg_encrypted;

        for (size_t i = 0; i < avg_encrypted.size(); i++)
        {
            for (size_t j = 0; j < avg_encrypted[i].size(); j++)
            {
                auto temp_de = encoder.decode_float64_polynomial_new(decryptor.decrypt_new(avg_encrypted[i][j]));
                auto filename = "conv2d_anonymous_im2col_avgpool_111_" + std::to_string(i) + "_" + std::to_string(j) + ".txt";
                writeToFile(temp_de, filename);

                evaluator.sub_plain_inplace(avg_encrypted[i][j], re_mask_encoded);

                temp_de = encoder.decode_float64_polynomial_new(decryptor.decrypt_new(avg_encrypted[i][j]));
                filename = "conv2d_anonymous_im2col_avgpool_222_" + std::to_string(i) + "_" + std::to_string(j) + ".txt";
                writeToFile(temp_de, filename);

                printf("i: %lu, j: %lu\n", i, j);
            }
        }

        vector<double> weight = read_file("../weights/model_fc_weight.txt");
        vector<double> bias = read_file("../weights/model_fc_bias.txt");
        Plain2d weight_encoded = help_10_64_1.encode_inputs_doubles(encoder, weight.data(), std::nullopt, scale);
        Plain2d bias_encoded = help_10_64_1.encode_outputs_doubles(encoder, bias.data(), std::nullopt, scale * scale);

        Cipher2d ret = help_10_64_1.matmul_reverse(evaluator, weight_encoded, avg_encrypted);

#ifdef DEBUG

        PRINT_2D_VECTOR_SHAPE(weight_encoded);
        PRINT_2D_VECTOR_SHAPE(avg_encrypted);
        PRINT_2D_VECTOR_SHAPE(ori_avg_encrypted);
        PRINT_2D_VECTOR_SHAPE(ret);

        auto temp_de = encoder.decode_float64_polynomial_new(decryptor.decrypt_new(ori_avg_encrypted[0][0]));
        auto filename = "conv2d_anonymous_im2col_avgpool_ori_avg_encrypted.txt";
        writeToFile(temp_de, filename);

        temp_de = encoder.decode_float64_polynomial_new(decryptor.decrypt_new(avg_encrypted[0][0]));
        filename = "conv2d_anonymous_im2col_avgpool_avg_encrypted.txt";
        writeToFile(temp_de, filename);

        temp_de = encoder.decode_float64_polynomial_new(decryptor.decrypt_new(ret[0][0]));
        filename = "conv2d_anonymous_im2col_avgpool_ret.txt";
        writeToFile(temp_de, filename);

        temp_de = encoder.decode_float64_polynomial_new(weight_encoded[0][0]);
        filename = "conv2d_anonymous_im2col_avgpool_weight_encoded.txt";
        writeToFile(temp_de, filename);
#endif

        ret.add_plain_inplace(evaluator, bias_encoded);

        vector<double> ret_decrypted = help_10_64_1.decrypt_outputs_doubles(encoder, decryptor, ret);
        writeToFile(ret_decrypted, "conv2d_anonymous_im2col_avgpool_fc.txt");
        for (double var : ret_decrypted)
        {
            cout << var << " ";
        }
        printf("\n");

        return ret_decrypted;
    }
};

