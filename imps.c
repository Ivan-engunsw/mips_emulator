////////////////////////////////////////////////////////////////////////
// Written by Ivan Lun Hui Chen (z5557064) on 2024-11-14.
// This program emulates the behaviour of MIPS instructions and memory processes
//

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>

#define BYTESIZE 8
#define FOUR_BYTES 4
#define MAGICNUMBER_INITIAL_SHIFT 24
#define NUM_REGISTER 32
#define IMMEDIATE_BYTES 2
#define DATA_SEGMENT 0x10010000
#define TWOS_COMPLEMENT_EIGHT_BITS 0x100
#define TWOS_COMPLEMENT_SIXTEEN_BITS 0x10000
#define MEMORY_ERROR_STRING_SIZE 5
#define IMPS_EXTENSION_LENGTH 4
#define MAX_FILE_SIZE 128
#define MAX_NUM_FILES 6
#define MAX_NUM_FILE_DESCRIPTORS 8

enum validMagicNumberBytes
{
    VALIDBYTE1 = 0x49,
    VALIDBYTE2 = 0x4d,
    VALIDBYTE3 = 0x50,
    VALIDBYTE4 = 0x53,
};

enum registerIndexes
{
    ZERO = 0,
    AT = 1,
    V0 = 2,
    V1 = 3,
    A0 = 4,
    A1 = 5,
    A2 = 6,
    A3 = 7,
    T0 = 8,
    T1 = 9,
    T2 = 10,
    T3 = 11,
    T4 = 12,
    T5 = 13,
    T6 = 14,
    T7 = 15,
    S0 = 16,
    S1 = 17,
    S2 = 18,
    S3 = 19,
    S4 = 20,
    S5 = 21,
    S6 = 22,
    S7 = 23,
    T8 = 24,
    T9 = 25,
    K0 = 26,
    K1 = 27,
    gp = 28,
    sp = 29,
    fp = 30,
    ra = 31,
};

enum shifts
{
    REGISTERSB_RIGHT_SHIFT = 21,
    REGISTERT_RIGHT_SHIFT = 16,
    REGISTERD_RIGHT_SHIFT = 11,
    LUI_SHIFT = 16,
    LEADING_BITS_SHIFT = 31
};

enum bitMasks
{
    MAGICNUMBER_MASK = 0xff,
    OPCODE_MASK = 0xfc000000,
    REGISTERSB_MASK = 0x03e00000,
    REGISTERT_MASK = 0x001f0000,
    REGISTERD_MASK = 0x0000f800,
    IMMEDIATE_MASK = 0x0000ffff,
    REMAININGBITS_MASK = 0x000007ff,
    SYSCALL_MASK = 0x0000003f,
    LEADING_BITS_MASK = 0x1,
    BYTE_MASK = 0x000000ff
};

enum opCodes
{
    ADDI = 0x20000000,
    ORI = 0x34000000,
    LUI = 0x3c000000,
    ADDIU = 0x24000000,
    MUL = 0x70000000,
    BEQ = 0x10000000,
    BNE = 0x14000000,
    LB = 0x80000000,
    LH = 0x84000000,
    LW = 0x8c000000,
    SB = 0xa0000000,
    SH = 0xa4000000,
    SW = 0xac000000
};

enum remainingBitsCode
{
    SYSCALL = 0xc,
    ADD = 0x20,
    CLO = 0x51,
    CLZ = 0x50,
    ADDU = 0x21,
    SLT = 0x2a
};

enum syscallCode
{
    PRINT_INT = 1,
    EXIT = 10,
    PRINT_CHAR = 11,
    PRINT_STRING = 4,
    READ_CHAR = 12,
    OPEN_FILE = 13,
    READ_FILE = 14,
    WRITE_FILE = 15,
    CLOSE_FILE = 16
};

enum memoryByteSize
{
    BYTE = 1,
    HALFWORD = 2,
    WORD = 4,
};

struct imps_file
{
    uint32_t num_instructions;
    uint32_t entry_point;
    uint32_t *instructions;
    uint32_t *debug_offsets;
    uint16_t memory_size;
    uint8_t *initial_data;
};

struct fileDescriptorInfo
{
    int flag;
    uint32_t address;
    int currentIndex;
};

struct file
{
    uint32_t address;
    char content[MAX_FILE_SIZE];
    int currentFileSize;
};

void read_imps_file(char *path, struct imps_file *executable);
void execute_imps(struct imps_file *executable, int trace_mode, 
                         char *path);
static void traceModePrint(struct imps_file *executable, char *path, 
                           int index);
static void printTraceInstruction(FILE *traceFile);
static void print_uint32_in_hexadecimal(FILE *stream, uint32_t value);
static void print_int32_in_decimal(FILE *stream, int32_t value);

static bool checkMagicNumberValidity(uint32_t magicNumber);
static uint32_t readLittleEndian32BitInt(FILE *inputStream);
static uint32_t *allocateAndFill32BitArray(struct imps_file *executable,
                                           FILE *inputStream);

static void instructionBitExtract(uint32_t instruction, uint32_t *registers,
                                  int *programCounter,
                                  struct imps_file *executable,
                                  int traceMode,
                                  struct fileDescriptorInfo *fileDescriptors,
                                  struct file *files);
static void execute_instruction(uint32_t *registers, uint32_t opCode,
                                uint32_t registerSB, uint32_t registerT,
                                uint32_t registerD, uint32_t immediate,
                                uint32_t remainingBits, uint32_t instruction,
                                int *programCounter,
                                struct imps_file *executable, int traceMode,
                                struct fileDescriptorInfo *fileDescriptors,
                                struct file *files);
static uint32_t negativeNumberSignExtension(uint32_t value, int numBytes);
static void additionOverflowCheck(uint32_t value1, uint32_t value2);
static void tracePrint(uint32_t registerPreviousvalue, uint32_t registerGiven,
                       uint32_t *registers, int traceMode);
static void branching(uint32_t immediate, int *programCounter);
static void opCodeZeroInstructions(uint32_t *registers, uint32_t registerSB,
                                   uint32_t registerT, uint32_t registerD,
                                   uint32_t immediate, uint32_t remainingBits,
                                   uint32_t instruction, int traceMode,
                                   struct imps_file *executable,
                                   struct fileDescriptorInfo *fileDescriptors,
                                   struct file *files);
static void syscall(uint32_t *registers, struct imps_file *executable,
                    struct fileDescriptorInfo *fileDescriptors,
                    struct file *files, int traceMode);
static void setFileDescriptor(int fileDescriptorIndex, uint32_t *registers,
                              struct fileDescriptorInfo *fileDescriptors,
                              struct file *files);
static struct file *findFileExistence(struct file *files, uint32_t fileToFind);
static void addFileExistence(struct file *files, uint32_t fileToAdd,
                             uint32_t *registers);
static void readFile(struct fileDescriptorInfo *fileDescriptors,
                     uint32_t *registers, struct imps_file *executable,
                     struct file *files);
static void writeFile(struct fileDescriptorInfo *fileDescriptors,
                      uint32_t *registers, struct imps_file *executable,
                      struct file *files);
static void updateFileSize(struct fileDescriptorInfo *fileDescriptors,
                      uint32_t *registers, struct file *fileRead);
static int countLeadingBits(uint32_t number, uint32_t bitWanted);
static void printBadInstructionError(uint32_t instruction);
static void executeMemoryInstructions(struct imps_file *executable,
                                      uint32_t *registers, uint32_t opCode,
                                      uint32_t registerSB, uint32_t registerT,
                                      uint32_t offset, uint32_t instruction,
                                      int traceMode);
static uint32_t loadMemory(struct imps_file *executable, uint32_t *registers,
                           uint32_t loadAddress, uint32_t offset, int size);
static void saveMemory(struct imps_file *executable, uint32_t *registers,
                       uint32_t loadAddress, uint32_t valueToSave,
                       uint32_t offset, int size);
static void printMemoryAccessError(uint32_t address, int size);

int main(int argc, char *argv[])
{
    char *pathname;
    int trace_mode = 0;

    if (argc == 2)
    {
        pathname = argv[1];
    }
    else if (argc == 3 && strcmp(argv[1], "-t") == 0)
    {
        trace_mode = 1;
        pathname = argv[2];
    }
    else
    {
        fprintf(stderr, "Usage: imps [-t] <executable>\n");
        exit(1);
    }

    struct imps_file executable = {0};
    read_imps_file(pathname, &executable);

    execute_imps(&executable, trace_mode, pathname);

    free(executable.debug_offsets);
    free(executable.instructions);
    free(executable.initial_data);

    return 0;
}

/**
 * Read an IMPS executable file from the file at `path` into `executable`.
 * Exits the program if the file can't be accessed or is not well-formed.
 */
void read_imps_file(char *path, struct imps_file *executable)
{
    // opens the file for reading
    FILE *inputStream = fopen(path, "r");
    if (inputStream == NULL)
    {
        perror(path);
        exit(EXIT_FAILURE);
    }

    uint32_t magicNumber = 0;
    // reads magicNumber by reading in four bytes consecutively
    for (int magicNumberIndex = 0; magicNumberIndex < FOUR_BYTES;
         magicNumberIndex++)
    {
        uint32_t readByte = fgetc(inputStream);
        magicNumber |= (readByte << (MAGICNUMBER_INITIAL_SHIFT -
                                     magicNumberIndex * BYTESIZE));
    }

    if (!checkMagicNumberValidity(magicNumber))
    {
        fclose(inputStream);
        fprintf(stderr, "Invalid IMPS file\n");
        exit(EXIT_FAILURE);
    }

    executable->num_instructions = readLittleEndian32BitInt(inputStream);
    executable->entry_point = readLittleEndian32BitInt(inputStream);
    executable->instructions = allocateAndFill32BitArray(executable,
                                                         inputStream);
    executable->debug_offsets = allocateAndFill32BitArray(executable,
                                                          inputStream);

    executable->memory_size = 0;
    // reads in the memory size as a little endian unsigned 16 bit integer
    for (int index = 0; index < 2; index++)
    {
        uint16_t readByte = fgetc(inputStream);
        executable->memory_size |= (readByte << index * BYTESIZE);
    }

    executable->initial_data = malloc(executable->memory_size *
                                      (sizeof(uint8_t)));
    for (int index = 0; index < executable->memory_size; index++)
    {
        uint8_t readByte = fgetc(inputStream);
        executable->initial_data[index] = readByte;
    }

    fclose(inputStream);
}

/**
 * Checks each byte of the magicNumber read against the valid bytes and returns
 * the validity.
 */
static bool checkMagicNumberValidity(uint32_t magicNumber)
{
    uint32_t magicNumberMask = MAGICNUMBER_MASK;
    for (int i = 0; i < FOUR_BYTES; i++)
    {
        uint8_t maskedMagicNumber = (magicNumber & magicNumberMask) 
                                    >> (i * BYTESIZE);
        magicNumberMask <<= BYTESIZE;
        if ((maskedMagicNumber != VALIDBYTE1) 
            && (maskedMagicNumber != VALIDBYTE2) 
            && (maskedMagicNumber != VALIDBYTE3) 
            && (maskedMagicNumber != VALIDBYTE4))
        {
            return false;
        }
    }
    return true;
}

/** reads a 32 bit unsigned integer that is of little endian format by storing
 *  the least significant byte first and shifting the position of the next byte
 * read to the left of the current byte.
 */
static uint32_t readLittleEndian32BitInt(FILE *inputStream)
{
    uint32_t result = 0;
    for (int index = 0; index < FOUR_BYTES; index++)
    {
        uint32_t readByte = fgetc(inputStream);
        result |= (readByte << index * BYTESIZE);
    }
    return result;
}

/**
 * reads in the instructions as unsigned 32 bit integers and storing it in an
 * array that stores all the instructions of the file.
 */
static uint32_t *allocateAndFill32BitArray(struct imps_file *executable,
                                           FILE *inputStream)
{
    uint32_t *newArray = malloc(executable->num_instructions *
                                sizeof(uint32_t));
    for (int i = 0; i < executable->num_instructions; i++)
    {
        newArray[i] = readLittleEndian32BitInt(inputStream);
    }
    return newArray;
}

/**
 * Executes an IMPS program
 */
void execute_imps(struct imps_file *executable, int trace_mode, char *path)
{
    uint32_t registers[NUM_REGISTER] = {0};

    struct fileDescriptorInfo fileDescriptors[MAX_NUM_FILE_DESCRIPTORS];
    for (int j = 0; j < MAX_NUM_FILE_DESCRIPTORS; j++)
    {
        //initialising fileDescriptors
        fileDescriptors[j].address = 0;
        fileDescriptors[j].flag = -1;
    }

    struct file files[MAX_NUM_FILES];
    for (int k = 0; k < MAX_NUM_FILES; k++)
    {
        //initialising files
        files[k].address = 0;
        files[k].currentFileSize = 0;
    }

    int i = executable->entry_point;
    while (true)
    {
        if (i >= executable->num_instructions)
        {
            fprintf(stderr,
                    "IMPS error: execution past the end of instructions\n");
            exit(EXIT_FAILURE);
        }

        if (trace_mode)
        {
            traceModePrint(executable, path, i);
        }

        instructionBitExtract(executable->instructions[i], registers, &i,
                              executable, trace_mode, fileDescriptors, files);
        i++;
    }
}

/**
 * Prints the traces of instruction under trace mode
 */
static void traceModePrint(struct imps_file *executable, char *path, 
                           int index) 
{
    //getting tracePath by replacing the imps file type to s file type
    char *tracePath = malloc((strlen(path) + 1) * sizeof(char));
    strncpy(tracePath, path, strlen(path) - IMPS_EXTENSION_LENGTH);
    tracePath[strlen(path) - IMPS_EXTENSION_LENGTH] = 's';
    tracePath[strlen(path) - (IMPS_EXTENSION_LENGTH - 1)] = '\0';

    FILE *traceFile = fopen(tracePath, "r");
    if (traceFile == NULL)
    {
        perror(tracePath);
        exit(EXIT_FAILURE);
    }

    int fseekSuccess = fseek(traceFile,
                                (long)executable->debug_offsets[index],
                                SEEK_SET);
    // ensures that the debug_offset does not cause the program to
    // print past the file
    if (fseekSuccess != -1)
    {
        printTraceInstruction(traceFile);
    }
    fclose(traceFile);
}

/**
 * prints the assembly form of the current instruction
 */
static void printTraceInstruction(FILE *traceFile) {
    int readByte = fgetc(traceFile);
    while ((readByte != '\n') && (readByte != EOF))
    {
        printf("%c", readByte);
        readByte = fgetc(traceFile);
    }

    if (readByte != EOF)
    {
        printf("\n");
    }
}

/**
 * extracts the opCode, registers s/b, t, d, the immediate value or remaining
 * bits out of the current instruction by using the corresponding bit masks and
 * shifts.
 */
static void instructionBitExtract(uint32_t instruction, uint32_t *registers,
                                  int *programCounter,
                                  struct imps_file *executable, int traceMode,
                                  struct fileDescriptorInfo *fileDescriptors,
                                  struct file *files)
{
    uint32_t opCode = instruction & OPCODE_MASK;
    uint32_t registerSB = (instruction & REGISTERSB_MASK) 
                            >> REGISTERSB_RIGHT_SHIFT;
    uint32_t registerT = (instruction & REGISTERT_MASK) 
                            >> REGISTERT_RIGHT_SHIFT;
    uint32_t registerD = (instruction & REGISTERD_MASK) 
                            >> REGISTERD_RIGHT_SHIFT;
    uint32_t immediate = (instruction & IMMEDIATE_MASK);
    uint32_t remainingBits = instruction & REMAININGBITS_MASK;

    execute_instruction(registers, opCode, registerSB, registerT, registerD,
                        immediate, remainingBits, instruction, programCounter,
                        executable, traceMode, fileDescriptors, files);
}

/**
 * executes the current instruction based off the opCode of the instruction
 */
static void execute_instruction(uint32_t *registers, uint32_t opCode,
                                uint32_t registerSB, uint32_t registerT,
                                uint32_t registerD, uint32_t immediate,
                                uint32_t remainingBits, uint32_t instruction,
                                int *programCounter, 
                                struct imps_file *executable, int traceMode,
                                struct fileDescriptorInfo *fileDescriptors,
                                struct file *files)
{
    uint32_t registerPreviousValue = registers[registerT];

    if (opCode == ADDI)
    {
        immediate = negativeNumberSignExtension(immediate, IMMEDIATE_BYTES);
        additionOverflowCheck(registers[registerSB], immediate);
        registers[registerT] = registers[registerSB] + immediate;
        tracePrint(registerPreviousValue, registerT, registers, traceMode);
    }
    else if (opCode == ORI)
    {
        registers[registerT] = registers[registerSB] | immediate;
        tracePrint(registerPreviousValue, registerT, registers, traceMode);
    }
    else if (opCode == LUI)
    {
        registers[registerT] = immediate << LUI_SHIFT;
        tracePrint(registerPreviousValue, registerT, registers, traceMode);
    }
    else if (opCode == 0)
    {
        opCodeZeroInstructions(registers, registerSB, registerT, registerD,
                               immediate, remainingBits, instruction,
                               traceMode, executable, fileDescriptors, files);
    }
    else if (opCode == ADDIU)
    {
        immediate = negativeNumberSignExtension(immediate, IMMEDIATE_BYTES);
        registers[registerT] = registers[registerSB] + immediate;
        tracePrint(registerPreviousValue, registerT, registers, traceMode);
    }
    else if (opCode == MUL)
    {
        registerPreviousValue = registers[registerD];
        registers[registerD] = registers[registerSB] * registers[registerT];
        tracePrint(registerPreviousValue, registerD, registers, traceMode);
    }
    else if (opCode == BEQ)
    {
        if (registers[registerSB] == registers[registerT])
        {
            branching(immediate, programCounter);
        }
    }
    else if (opCode == BNE)
    {
        if (registers[registerSB] != registers[registerT])
        {
            branching(immediate, programCounter);
        }
    }
    else if (((opCode & ((uint32_t)LEADING_BITS_MASK << LEADING_BITS_SHIFT)) 
                >> LEADING_BITS_SHIFT) == 1)
    {
        // memory instructions
        executeMemoryInstructions(executable, registers, opCode, registerSB,
                                  registerT, immediate, instruction, 
                                  traceMode);
    }
    else
    {
        printBadInstructionError(instruction);
    }

    // ensuring that $0 remains 0
    registers[ZERO] = ZERO;
}

/**
 * converts the value given into its negative representation by using two's
 * complement with the relevant number of bits.
 */
static uint32_t negativeNumberSignExtension(uint32_t value, int numBytes)
{
    if ((value >> (numBytes * BYTESIZE - 1)) & 1)
    {
        if (numBytes == BYTE)
        {
            value -= TWOS_COMPLEMENT_EIGHT_BITS;
        }
        else if (numBytes == HALFWORD)
        {
            value -= TWOS_COMPLEMENT_SIXTEEN_BITS;
        }
    }
    return value;
}

/**
 * checks if the addition of the two values will lead to an overflow by checking
 * if the result's sign has changed as as result of the addition
 */
static void additionOverflowCheck(uint32_t value1, uint32_t value2)
{
    uint32_t signBitMask = LEADING_BITS_MASK;
    signBitMask <<= LEADING_BITS_SHIFT;

    // if the two values are of the same sign, execute the check
    if ((value1 & signBitMask) == (value2 & signBitMask))
    {
        uint32_t result = value1 + value2;
        // if the result's sign is different to the values' sign, output error
        if ((result & signBitMask) != (value1 & signBitMask))
        {
            fprintf(stderr, "IMPS error: addition would overflow\n");
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * prints the change in value in the registers if it has changed
 */
static void tracePrint(uint32_t registerPreviousvalue, uint32_t registerGiven,
                       uint32_t *registers, int traceMode)
{
    char *registerNames[32] = {"zer0", "at", "v0", "v1", "a0", "a1", "a2", 
                               "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
                               "t7", "s0", "s1", "s2", "s3", "s4", "s5", "s6", 
                               "s7", "t8", "t9", "k0", "k1", "gp", "sp", "fp", 
                               "ra"};

    if ((traceMode) && (registers[registerGiven] != registerPreviousvalue))
    {
        printf("   $%s: ", registerNames[registerGiven]);
        print_uint32_in_hexadecimal(stdout, registerPreviousvalue);
        printf(" -> ");
        print_uint32_in_hexadecimal(stdout, registers[registerGiven]);
        printf("\n");
    }
}

/**
 * branches to the next instruction with the offset given by increasing the
 * program counter
 */
static void branching(uint32_t immediate, int *programCounter)
{
    immediate = negativeNumberSignExtension(immediate, IMMEDIATE_BYTES);
    // minus one because we increment in execute_imps after this function
    *programCounter += (immediate - 1);
}

/**
 * executes the instructions with the opcode of 000000
 */
static void opCodeZeroInstructions(uint32_t *registers, uint32_t registerSB,
                                   uint32_t registerT, uint32_t registerD,
                                   uint32_t immediate, uint32_t remainingBits,
                                   uint32_t instruction, int traceMode,
                                   struct imps_file *executable,
                                   struct fileDescriptorInfo *fileDescriptors,
                                   struct file *files)
{
    uint32_t registerPreviousValue = registers[registerD];

    if ((instruction & SYSCALL_MASK) == SYSCALL)
    {
        syscall(registers, executable, fileDescriptors, files, traceMode);
    }
    else if (remainingBits == ADD)
    {
        additionOverflowCheck(registers[registerSB], registers[registerT]);
        registers[registerD] = registers[registerSB] + registers[registerT];
        tracePrint(registerPreviousValue, registerD, registers, traceMode);
    }
    else if (remainingBits == CLO)
    {
        registers[registerD] = countLeadingBits(registers[registerSB], 1);
        tracePrint(registerPreviousValue, registerD, registers, traceMode);
    }
    else if (remainingBits == CLZ)
    {
        registers[registerD] = countLeadingBits(registers[registerSB], 0);
        tracePrint(registerPreviousValue, registerD, registers, traceMode);
    }
    else if (remainingBits == ADDU)
    {
        registers[registerD] = registers[registerSB] + registers[registerT];
        tracePrint(registerPreviousValue, registerD, registers, traceMode);
    }
    else if (remainingBits == SLT)
    {
        registers[registerD] = (int32_t)registers[registerSB] <
                               (int32_t)registers[registerT];
        tracePrint(registerPreviousValue, registerD, registers, traceMode);
    }
}

/**
 * executes the syscall functions
 */
static void syscall(uint32_t *registers, struct imps_file *executable,
                    struct fileDescriptorInfo *fileDescriptors,
                    struct file *files, int traceMode)
{
    uint32_t v0 = registers[V0];

    if (v0 == PRINT_INT)
    {
        print_int32_in_decimal(stdout, (int32_t)registers[A0]);
    }
    else if (v0 == EXIT)
    {
        exit(EXIT_SUCCESS);
    }
    else if (v0 == PRINT_CHAR)
    {
        putchar((int)registers[A0]);
    }
    else if (v0 == PRINT_STRING)
    {
        uint32_t offset = 0;
        char character = loadMemory(executable, registers, registers[A0], 
        offset, BYTE);

        while (character != '\0')
        {
            fprintf(stdout, "%c", character);
            offset++;
            character = loadMemory(executable, registers, registers[A0], 
                                   offset, BYTE);
        }
    }
    else if (v0 == READ_CHAR)
    {
        int character = getchar();
        registers[V0] = character;
    }
    else if (v0 == OPEN_FILE)
    {
        uint32_t registerPreviousValue = registers[V0];
        int fileDescriptorIndex = 0;

        //gets the next available file descriptor index
        while ((fileDescriptorIndex < MAX_NUM_FILE_DESCRIPTORS) &&
               (fileDescriptors[fileDescriptorIndex].address != 0))
        {
            fileDescriptorIndex++;
        }

        //if the available file descriptor index is invalid
        if (fileDescriptorIndex >= MAX_NUM_FILE_DESCRIPTORS)
        {
            registers[V0] = -1;
        }
        else
        {
            setFileDescriptor(fileDescriptorIndex, registers, fileDescriptors,
                              files);
        }

        tracePrint(registerPreviousValue, V0, registers, traceMode);
    }
    else if (v0 == READ_FILE)
    {
        uint32_t registerPreviousValue = registers[V0];
        readFile(fileDescriptors, registers, executable, files);
        tracePrint(registerPreviousValue, V0, registers, traceMode);
    }
    else if (v0 == WRITE_FILE)
    {
        uint32_t registerPreviousValue = registers[V0];
        writeFile(fileDescriptors, registers, executable, files);
        tracePrint(registerPreviousValue, V0, registers, traceMode);
    }
    else if (v0 == CLOSE_FILE)
    {
        uint32_t registerPreviousValue = registers[V0];

        //if the fileDescriptor given is invalid
        if ((registers[A0] < 0) || (registers[A0] >= MAX_NUM_FILE_DESCRIPTORS) 
            || (fileDescriptors[registers[A0]].address == 0))
        {
            registers[V0] = -1;
        }
        else
        {
            registers[V0] = 0;
            //resets the fileDescriptor so it can be allocated again
            fileDescriptors[registers[A0]].flag = -1;
            fileDescriptors[registers[A0]].currentIndex = 0;
            fileDescriptors[registers[A0]].address = 0;
        }

        tracePrint(registerPreviousValue, V0, registers, traceMode);
    }
    else
    {
        fprintf(stderr, "IMPS error: bad syscall number\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * sets the new file descriptor with all the information that it needs
 */
static void setFileDescriptor(int fileDescriptorIndex, uint32_t *registers,
                              struct fileDescriptorInfo *fileDescriptors,
                              struct file *files)
{
    //if the flag is not read or write
    if ((registers[A1] != 0) && (registers[A1] != 1))
    {
        registers[V0] = -1;
    }
    else
    {
        if (registers[A1] == 1)
        {
            //check if the file exists, if not create it
            if (findFileExistence(files, registers[A0]) == NULL)
            {
                addFileExistence(files, registers[A0], registers);
            }
            fileDescriptors[fileDescriptorIndex].address = registers[A0];
            fileDescriptors[fileDescriptorIndex].flag = registers[A1];
            fileDescriptors[fileDescriptorIndex].currentIndex = 0;
            registers[V0] = fileDescriptorIndex;
        }
        else
        {
            if (findFileExistence(files, registers[A0]) == NULL)
            {
                registers[V0] = -1;
            }
            else
            {
                fileDescriptors[fileDescriptorIndex].address = registers[A0];
                fileDescriptors[fileDescriptorIndex].flag = registers[A1];
                fileDescriptors[fileDescriptorIndex].currentIndex = 0;
                registers[V0] = fileDescriptorIndex;
            }
        }
    }
}

/**
 * searches through the files array to find if the file given exists. Returns
 * the file if it does and NULL if not
 */
static struct file *findFileExistence(struct file *files, uint32_t fileToFind)
{
    for (int i = 0; i < MAX_NUM_FILES; i++)
    {
        if (files[i].address == fileToFind)
        {
            return &files[i];
        }
    }
    return NULL;
}

/**
 * adds a new file into the files array if there is still space for the file
 * in the array
 */
static void addFileExistence(struct file *files, uint32_t fileToAdd,
                             uint32_t *registers)
{
    int i = 0;
    // gets the next available file index
    while ((i < MAX_NUM_FILES) && (files[i].address != 0))
    {
        i++;
    }

    if (i < MAX_NUM_FILES)
    {
        files[i].address = fileToAdd;
    }
    else
    {
        registers[V0] = -1;
    }
}

/**
 * Reads the content of a file up to $a2 many bytes into the $a1 buffer
 */
static void readFile(struct fileDescriptorInfo *fileDescriptors,
                     uint32_t *registers, struct imps_file *executable,
                     struct file *files)
{
    if (fileDescriptors[registers[A0]].flag != 0)
    {
        registers[V0] = -1;
    }
    else
    {
        int i = 0;
        struct file *fileRead = findFileExistence(files, 
            fileDescriptors[registers[A0]].address);
        while (i < (int)registers[A2])
        {
            int index = fileDescriptors[registers[A0]].currentIndex;
            if ((fileRead == NULL) || (index >= MAX_FILE_SIZE))
            {
                registers[V0] = -1;
                break;
            }

            // there is no more in the file to be read
            if (index >= fileRead->currentFileSize)
            {
                break;
            }

            if (i < fileRead->currentFileSize)
            {
                uint32_t valueToSave = fileRead->content[index];
                saveMemory(executable, registers, registers[A1], valueToSave, 
                           i, BYTE);
                i++;
                fileDescriptors[registers[A0]].currentIndex++;
            }
            else
            {
                break;
            }
        }

        if (registers[V0] != -1)
        {
            registers[V0] = i;
        }
    }
}

/**
 * writes to the file from the $a1 buffer up to $a2 many bytes
 */
static void writeFile(struct fileDescriptorInfo *fileDescriptors,
                      uint32_t *registers, struct imps_file *executable,
                      struct file *files)
{
    if (fileDescriptors[registers[A0]].flag != 1)
    {
        registers[V0] = -1;
    }
    else
    {
        int i = 0;
        struct file *fileRead = findFileExistence(files, 
            fileDescriptors[registers[A0]].address);
        while (i < (int)registers[A2])
        {
            uint32_t valueToSave = loadMemory(executable, registers,
                                              registers[A1], i, BYTE);
            valueToSave = negativeNumberSignExtension(valueToSave, BYTE);
            int index = fileDescriptors[registers[A0]].currentIndex;

            if ((fileRead == NULL) || (index >= MAX_FILE_SIZE))
            {
                registers[V0] = -1;
                break;
            }

            if (valueToSave != EOF)
            {
                fileRead->content[index] = valueToSave;
                i++;
                updateFileSize(fileDescriptors, registers, fileRead);
                fileDescriptors[registers[A0]].currentIndex++;
            }
            else
            {
                break;
            }
        }

        if (registers[V0] != -1)
        {
            registers[V0] = i;
        }
    }
}

/**
 * updating the file size as more bytes are written into the file
 */
static void updateFileSize(struct fileDescriptorInfo *fileDescriptors,
                      uint32_t *registers, struct file *fileRead) 
{
    if (fileDescriptors[registers[A0]].currentIndex 
        >= fileRead->currentFileSize)
    {
        fileRead->currentFileSize++;
    }
}

/**
 * counts the leading bits by finding counting the bits until it the bit is
 * different to the bitWanted
 */
static int countLeadingBits(uint32_t number, uint32_t bitWanted)
{
    uint32_t bitMask = LEADING_BITS_MASK;
    bitMask <<= LEADING_BITS_SHIFT;
    int count = 0;
    int shift = LEADING_BITS_SHIFT;

    while (shift >= 0 && (((number & bitMask) >> shift) == bitWanted))
    {
        count++;
        shift--;
        bitMask >>= 1;
    }

    return count;
}

/**
 * prints an error message if the instruction does not match with any current
 * instruction checks
 */
static void printBadInstructionError(uint32_t instruction)
{
    fprintf(stderr, "IMPS error: bad instruction ");
    print_uint32_in_hexadecimal(stderr, instruction);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

/**
 * executes instructions that involve accessing and storing in memory
 */
static void executeMemoryInstructions(struct imps_file *executable,
                                      uint32_t *registers, uint32_t opCode,
                                      uint32_t registerSB, uint32_t registerT,
                                      uint32_t offset, uint32_t instruction,
                                      int traceMode)
{
    uint32_t registerPreviousValue = registers[registerT];

    if (opCode == LB)
    {
        registers[registerT] = loadMemory(executable, registers, 
                                          registers[registerSB], offset, BYTE);
        tracePrint(registerPreviousValue, registerT, registers, traceMode);
    }
    else if (opCode == LH)
    {
        registers[registerT] = loadMemory(executable, registers, 
                                          registers[registerSB], offset, 
                                          HALFWORD);
        tracePrint(registerPreviousValue, registerT, registers, traceMode);
    }
    else if (opCode == LW)
    {
        registers[registerT] = loadMemory(executable, registers, 
                                          registers[registerSB], offset, WORD);
        tracePrint(registerPreviousValue, registerT, registers, traceMode);
    }
    else if (opCode == SB)
    {
        saveMemory(executable, registers, registers[registerSB], 
                   registers[registerT], offset, BYTE);
    }
    else if (opCode == SH)
    {
        saveMemory(executable, registers, registers[registerSB], 
                   registers[registerT], offset,
                   HALFWORD);
    }
    else if (opCode == SW)
    {
        saveMemory(executable, registers, registers[registerSB], 
                   registers[registerT], offset, WORD);
    }
    else
    {
        printBadInstructionError(instruction);
    }
}

/**
 * retrieves the values that is stored in the given address as calculated with
 * the given offset
 */
static uint32_t loadMemory(struct imps_file *executable, uint32_t *registers,
                           uint32_t loadAddress, uint32_t offset, int size)
{
    uint32_t finalAddress = loadAddress + offset;

    // check if address is out of bounds
    if (finalAddress < DATA_SEGMENT ||
        finalAddress >= (DATA_SEGMENT + executable->memory_size))
    {
        printMemoryAccessError(finalAddress, size);
    }

    uint32_t result = 0;
    uint32_t currentAddressIndex = finalAddress - DATA_SEGMENT;

    // check if the address is aligned
    if (finalAddress % size == 0)
    {
        for (int i = 0; i < size; i++)
        {
            uint32_t readByte = 
                executable->initial_data[currentAddressIndex + i];
            result |= (readByte << i * BYTESIZE);
        }
    }
    else
    {
        printMemoryAccessError(finalAddress, size);
    }

    result = negativeNumberSignExtension(result, size);
    return result;
}

/**
 * saves the value into the given memory as calculated with the given offset
 */
static void saveMemory(struct imps_file *executable, uint32_t *registers,
                       uint32_t loadAddress, uint32_t valueToSave,
                       uint32_t offset, int size)
{
    uint32_t finalAddress = loadAddress + offset;

    // check if address is out of bounds
    if (finalAddress < DATA_SEGMENT ||
        finalAddress >= (DATA_SEGMENT + executable->memory_size))
    {
        printMemoryAccessError(finalAddress, size);
    }

    uint32_t currentAddressIndex = finalAddress - DATA_SEGMENT;

    // check if address is aligned
    if (finalAddress % size == 0)
    {
        for (int i = 0; i < size; i++)
        {
            uint32_t byteMask = BYTE_MASK;
            byteMask <<= (i * BYTESIZE);
            executable->initial_data[currentAddressIndex + i] =
                ((valueToSave & byteMask) >> (i * BYTESIZE));
        }
    }
    else
    {
        printMemoryAccessError(finalAddress, size);
    }
}

/**
 * prints an error message for incorrect memory access
 */
static void printMemoryAccessError(uint32_t address, int size)
{
    char memorySize[MEMORY_ERROR_STRING_SIZE] = "byte";
    if (size == HALFWORD)
    {
        strcpy(memorySize, "half");
    }
    else if (size == WORD)
    {
        strcpy(memorySize, "word");
    }
    fprintf(stderr, "IMPS error: bad address for %s access: ", memorySize);
    print_uint32_in_hexadecimal(stderr, address);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

// Printing out exact-width integers in a portable way is slightly tricky,
// since we can't assume that a uint32_t is an unsigned int or that a
// int32_t is an int. So we can't just use %x or %d. A solution is to use
// printf format specifiers from the <inttypes.h> header. The following two
// functions are provided for your convenience so that you just call them
// without worring about <inttypes.h>, although you don't have to use use them.

// Print out a 32 bit integer in hexadecimal, including the leading `0x`.
//
// @param stream The file stream to output to.
// @param value The 32 bit integer to output.
void print_uint32_in_hexadecimal(FILE *stream, uint32_t value)
{
    fprintf(stream, "0x%08" PRIx32, value);
}

// Print out a signed 32 bit integer in decimal.
//
// @param stream The file stream to output to.
// @param value The 32 bit integer to output.
void print_int32_in_decimal(FILE *stream, int32_t value)
{
    fprintf(stream, "%" PRIi32, value);
}
