#include <Kokkos_Core.hpp>
#include <fstream>
#include <chrono>
//#include <filesystem>
#include <cstdio>
#include <string>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>

using std::ifstream;
using std::ios;
using std::ofstream;
using std::string;

double duration(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char **argv)
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    Kokkos::initialize(argc, argv);
    {
        string filename = "HK-7_left_H6D-400c-MS.bmp";
        // string filename = "HK-7_left_H6D-400c-MS_screw.bmp";
        // std::uintmax_t filesize = std::filesystem::file_size(filename);
        // printf("The file size is %ju\n", filesize);

        // Open File
        ifstream fin(filename, ios::in | ios::binary);
        if (!fin.is_open())
        {
            printf("File not opened\n");
            return -1;
        }
        // The first 14 bytes are the header, containing four values.  Get those four values.
        char header[2];
        uint32_t filesize;
        uint32_t dummy;
        uint32_t offset;
        fin.read(header, 2);
        fin.read((char *)&filesize, 4);
        fin.read((char *)&dummy, 4);
        fin.read((char *)&offset, 4);
        printf("header: %c%c\n", header[0], header[1]);
        printf("filesize: %u\n", filesize);
        printf("dummy %u\n", dummy);
        printf("offset: %u\n", offset);
        int32_t sizeOfHeader;
        int32_t width;
        int32_t height;
        fin.read((char *)&sizeOfHeader, 4);
        fin.read((char *)&width, 4);
        fin.read((char *)&height, 4);
        printf("The width: %d\n", width);
        printf("The height: %d\n", height);
        uint16_t numColorPanes;
        uint16_t numBitsPerPixel;
        fin.read((char *)&numColorPanes, 2);
        fin.read((char *)&numBitsPerPixel, 2);
        printf("The number of bits per pixel: %u\n", numBitsPerPixel);
        if (numBitsPerPixel == 24)
        {
            printf("This bitmap uses rgb, where the first byte is blue, second byte is green, third byte is red.\n");
        }
        //uint32_t rowSize = (numBitsPerPixel * width + 31) / 32 * 4;
        //printf("Each row in the image requires %u bytes\n", rowSize);

        // Jump to offset where the bitmap pixel data starts
        fin.seekg(offset, ios::beg);

        // Read the data part of the file
        unsigned char *h_buffer = new unsigned char[filesize - offset];
        fin.read((char *)h_buffer, filesize - offset);
        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point end;
        printf("The first pixel is located in the bottom left.  Its blue/green/red values are (%u, %u, %u)\n", h_buffer[0], h_buffer[1], h_buffer[2]);
        printf("The second pixel is to the right.  Its blue/green/red values are (%u, %u, %u)\n", h_buffer[3], h_buffer[4], h_buffer[5]);

        // TODO: Read the image into Kokkos views
        start = std::chrono::high_resolution_clock::now();
        Kokkos::View<int **, Kokkos::LayoutRight> blueValues("blueValues", height, width);
        Kokkos::View<int **, Kokkos::LayoutRight> greenValues("greenValues", height, width);
        Kokkos::View<int **, Kokkos::LayoutRight> redValues("redValues", height, width);
        Kokkos::View<int **, Kokkos::LayoutRight> blueValuesOut("blueValues", height, width);
        Kokkos::View<int **, Kokkos::LayoutRight> greenValuesOut("greenValues", height, width);
        Kokkos::View<int **, Kokkos::LayoutRight> redValuesOut("redValues", height, width);

        int h_buffer_index = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                blueValues(i, j) = h_buffer[h_buffer_index + 0];
                greenValues(i, j) = h_buffer[h_buffer_index + 1];
                redValues(i, j) = h_buffer[h_buffer_index + 2];
                h_buffer_index += 3;
            }
        }

        end = std::chrono::high_resolution_clock::now();
        printf("Time loading image -%g ms\n", duration(start, end));

        // TODO: Perform the blurring
        start = std::chrono::high_resolution_clock::now();
        height = height / world_size;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<>(0, height * width), KOKKOS_LAMBDA(const int n) {
                int i = n / width + (height * world_rank);
                int j = n % width;

                int blueSum = 0;
                blueSum += ((i - 2 * j - 2) < 0) ? 0 : blueValues(i - 2, j - 2) * 1;
                blueSum += ((i - 2 * j - 1) < 0) ? 0 : blueValues(i - 2, j - 1) * 4;
                blueSum += ((i - 2 * j - 0) < 0) ? 0 : blueValues(i - 2, j - 0) * 7;
                blueSum += ((i - 2 * j + 1) < 0) ? 0 : blueValues(i - 2, j + 1) * 4;
                blueSum += ((i - 2 * j + 2) < 0) ? 0 : blueValues(i - 2, j + 2) * 1;
                blueSum += ((i - 1 * j - 2) < 0) ? 0 : blueValues(i - 1, j - 2) * 4;
                blueSum += ((i - 1 * j - 1) < 0) ? 0 : blueValues(i - 1, j - 1) * 16;
                blueSum += ((i - 1 * j - 0) < 0) ? 0 : blueValues(i - 1, j - 0) * 26;
                blueSum += ((i - 1 * j + 1) < 0) ? 0 : blueValues(i - 1, j + 1) * 16;
                blueSum += ((i - 1 * j + 2) < 0) ? 0 : blueValues(i - 1, j + 2) * 4;
                blueSum += ((i - 0 * j - 2) < 0) ? 0 : blueValues(i - 0, j - 2) * 7;
                blueSum += ((i - 0 * j - 1) < 0) ? 0 : blueValues(i - 0, j - 1) * 26;
                blueSum += ((i - 0 * j - 0) < 0) ? 0 : blueValues(i - 0, j - 0) * 41;
                blueSum += ((i - 0 * j + 1) < 0) ? 0 : blueValues(i - 0, j + 1) * 26;
                blueSum += ((i - 0 * j + 2) < 0) ? 0 : blueValues(i - 0, j + 2) * 7;
                blueSum += ((i + 1 * j - 2) < 0) ? 0 : blueValues(i + 1, j - 2) * 4;
                blueSum += ((i + 1 * j - 1) < 0) ? 0 : blueValues(i + 1, j - 1) * 16;
                blueSum += ((i + 1 * j - 0) < 0) ? 0 : blueValues(i + 1, j - 0) * 26;
                blueSum += ((i + 1 * j + 1) < 0) ? 0 : blueValues(i + 1, j + 1) * 16;
                blueSum += ((i + 1 * j + 2) < 0) ? 0 : blueValues(i + 1, j + 2) * 4;
                blueSum += ((i + 2 * j - 2) < 0) ? 0 : blueValues(i + 2, j - 2) * 1;
                blueSum += ((i + 2 * j - 1) < 0) ? 0 : blueValues(i + 2, j - 1) * 4;
                blueSum += ((i + 2 * j - 0) < 0) ? 0 : blueValues(i + 2, j - 0) * 7;
                blueSum += ((i + 2 * j + 1) < 0) ? 0 : blueValues(i + 2, j + 1) * 4;
                blueSum += ((i + 2 * j + 2) < 0) ? 0 : blueValues(i + 2, j + 2) * 1;
                int bluePixelValue = blueSum / 273;

                int greenSum = 0;
                greenSum += ((i - 2 * j - 2) < 0) ? 0 : greenValues(i - 2, j - 2) * 1;
                greenSum += ((i - 2 * j - 1) < 0) ? 0 : greenValues(i - 2, j - 1) * 4;
                greenSum += ((i - 2 * j - 0) < 0) ? 0 : greenValues(i - 2, j - 0) * 7;
                greenSum += ((i - 2 * j + 1) < 0) ? 0 : greenValues(i - 2, j + 1) * 4;
                greenSum += ((i - 2 * j + 2) < 0) ? 0 : greenValues(i - 2, j + 2) * 1;
                greenSum += ((i - 1 * j - 2) < 0) ? 0 : greenValues(i - 1, j - 2) * 4;
                greenSum += ((i - 1 * j - 1) < 0) ? 0 : greenValues(i - 1, j - 1) * 16;
                greenSum += ((i - 1 * j - 0) < 0) ? 0 : greenValues(i - 1, j - 0) * 26;
                greenSum += ((i - 1 * j + 1) < 0) ? 0 : greenValues(i - 1, j + 1) * 16;
                greenSum += ((i - 1 * j + 2) < 0) ? 0 : greenValues(i - 1, j + 2) * 4;
                greenSum += ((i - 0 * j - 2) < 0) ? 0 : greenValues(i - 0, j - 2) * 7;
                greenSum += ((i - 0 * j - 1) < 0) ? 0 : greenValues(i - 0, j - 1) * 26;
                greenSum += ((i - 0 * j - 0) < 0) ? 0 : greenValues(i - 0, j - 0) * 41;
                greenSum += ((i - 0 * j + 1) < 0) ? 0 : greenValues(i - 0, j + 1) * 26;
                greenSum += ((i - 0 * j + 2) < 0) ? 0 : greenValues(i - 0, j + 2) * 7;
                greenSum += ((i + 1 * j - 2) < 0) ? 0 : greenValues(i + 1, j - 2) * 4;
                greenSum += ((i + 1 * j - 1) < 0) ? 0 : greenValues(i + 1, j - 1) * 16;
                greenSum += ((i + 1 * j - 0) < 0) ? 0 : greenValues(i + 1, j - 0) * 26;
                greenSum += ((i + 1 * j + 1) < 0) ? 0 : greenValues(i + 1, j + 1) * 16;
                greenSum += ((i + 1 * j + 2) < 0) ? 0 : greenValues(i + 1, j + 2) * 4;
                greenSum += ((i + 2 * j - 2) < 0) ? 0 : greenValues(i + 2, j - 2) * 1;
                greenSum += ((i + 2 * j - 1) < 0) ? 0 : greenValues(i + 2, j - 1) * 4;
                greenSum += ((i + 2 * j - 0) < 0) ? 0 : greenValues(i + 2, j - 0) * 7;
                greenSum += ((i + 2 * j + 1) < 0) ? 0 : greenValues(i + 2, j + 1) * 4;
                greenSum += ((i + 2 * j + 2) < 0) ? 0 : greenValues(i + 2, j + 2) * 1;
                int greenPixelValue = (greenSum / 273);

                int redSum = 0;
                redSum += ((i - 2 * j - 2) < 0) ? 0 : redValues(i - 2, j - 2) * 1;
                redSum += ((i - 2 * j - 1) < 0) ? 0 : redValues(i - 2, j - 1) * 4;
                redSum += ((i - 2 * j - 0) < 0) ? 0 : redValues(i - 2, j - 0) * 7;
                redSum += ((i - 2 * j + 1) < 0) ? 0 : redValues(i - 2, j + 1) * 4;
                redSum += ((i - 2 * j + 2) < 0) ? 0 : redValues(i - 2, j + 2) * 1;
                redSum += ((i - 1 * j - 2) < 0) ? 0 : redValues(i - 1, j - 2) * 4;
                redSum += ((i - 1 * j - 1) < 0) ? 0 : redValues(i - 1, j - 1) * 16;
                redSum += ((i - 1 * j - 0) < 0) ? 0 : redValues(i - 1, j - 0) * 26;
                redSum += ((i - 1 * j + 1) < 0) ? 0 : redValues(i - 1, j + 1) * 16;
                redSum += ((i - 1 * j + 2) < 0) ? 0 : redValues(i - 1, j + 2) * 4;
                redSum += ((i - 0 * j - 2) < 0) ? 0 : redValues(i - 0, j - 2) * 7;
                redSum += ((i - 0 * j - 1) < 0) ? 0 : redValues(i - 0, j - 1) * 26;
                redSum += ((i - 0 * j - 0) < 0) ? 0 : redValues(i - 0, j - 0) * 41;
                redSum += ((i - 0 * j + 1) < 0) ? 0 : redValues(i - 0, j + 1) * 26;
                redSum += ((i - 0 * j + 2) < 0) ? 0 : redValues(i - 0, j + 2) * 7;
                redSum += ((i + 1 * j - 2) < 0) ? 0 : redValues(i + 1, j - 2) * 4;
                redSum += ((i + 1 * j - 1) < 0) ? 0 : redValues(i + 1, j - 1) * 16;
                redSum += ((i + 1 * j - 0) < 0) ? 0 : redValues(i + 1, j - 0) * 26;
                redSum += ((i + 1 * j + 1) < 0) ? 0 : redValues(i + 1, j + 1) * 16;
                redSum += ((i + 1 * j + 2) < 0) ? 0 : redValues(i + 1, j + 2) * 4;
                redSum += ((i + 2 * j - 2) < 0) ? 0 : redValues(i + 2, j - 2) * 1;
                redSum += ((i + 2 * j - 1) < 0) ? 0 : redValues(i + 2, j - 1) * 4;
                redSum += ((i + 2 * j - 0) < 0) ? 0 : redValues(i + 2, j - 0) * 7;
                redSum += ((i + 2 * j + 1) < 0) ? 0 : redValues(i + 2, j + 1) * 4;
                redSum += ((i + 2 * j + 2) < 0) ? 0 : redValues(i + 2, j + 2) * 1;
                int redPixelValue = redSum / 273;

                // blueValuesOut(i, j) = 0;
                // greenValuesOut(i, j) = 0;
                // redValuesOut(i, j) = 0;
                blueValuesOut(i, j) = bluePixelValue;
                greenValuesOut(i, j) = greenPixelValue;
                redValuesOut(i, j) = redPixelValue;
            });
        end = std::chrono::high_resolution_clock::now();
        printf("Time processing image -%g ms\n", duration(start, end));

        // TODO: Verification
        printf("The red, green, blue at (8353, 9111) (origin bottom left) is (%d, %d, %d)\n", redValuesOut(8353, 9111), greenValuesOut(8353, 9111), blueValuesOut(8353, 9111));
        printf("The red, green, blue at (8351, 9113) (origin bottom left) is (%d, %d, %d)\n", redValuesOut(8351, 9113), greenValuesOut(8351, 9113), blueValuesOut(8351, 9113));
        printf("The red, green, blue at (6352, 15231) (origin bottom left) is (%d, %d, %d)\n", redValuesOut(6352, 15231), greenValuesOut(6352, 15231), blueValuesOut(6352, 15231));
        printf("The red, green, blue at (10559, 10611) (origin bottom left) is (%d, %d, %d)\n", redValuesOut(10559, 10611), greenValuesOut(10559, 10611), blueValuesOut(10559, 10611));
        printf("The red, green, blue at (10818, 20226) (origin bottom left) is (%d, %d, %d)\n", redValuesOut(10818, 20226), greenValuesOut(10818, 20226), blueValuesOut(10818, 20226));

        //Print out to file output.bmp
        string outputFile = "output" + std::to_string(world_rank) + ".bmp";
        ofstream fout;
        fout.open(outputFile, ios::binary);

        // Copy of the old headers into the new output
        fin.seekg(0, ios::beg);
        // Read the data part of the file
        char *headers = new char[offset];
        fin.read(headers, offset);
        fout.seekp(0, ios::beg);
        fout.write(headers, offset);
        delete[] headers;
        fout.seekp(offset, ios::beg);

        // TODO: Copy out the rest of the view to file(hint, use fout.put())
        start = std::chrono::high_resolution_clock::now();
        height = height * world_size;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                fout.put(blueValuesOut(i, j));
                fout.put(greenValuesOut(i, j));
                fout.put(redValuesOut(i, j));
            }
        }
        end = std::chrono::high_resolution_clock::now();
        printf("Time copy back to file -%g ms\n", duration(start, end));
        fout.close();
    }
    Kokkos::finalize();
    MPI_Finalize();
}
