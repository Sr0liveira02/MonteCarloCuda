# Compiler
CC = nvcc

# Optimization Flags
CFLAGS = -O3
INC    = -I/usr/include/hdf5/serial # both of this could be removed in case of knowing at u are doing
LIBDIR = -L/usr/lib/x86_64-linux-gnu #
LIBS   = -lhdf5_serial

# Executable Name
TARGET = m_exp
SRC = matrix_init.cu matrix_utils.cu main.cu AliasTable.cu m_exp.cu
OBJ = $(SRC:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LIBDIR) $(LIBS) -o $(TARGET) -lm

%.o: %.cu
	$(CC) $(CFLAGS) $(INC) -c $< -o $@ -lm  # Link `-lm` at this stage

# Clean
clean:
	rm -f $(OBJ) $(TARGET)
