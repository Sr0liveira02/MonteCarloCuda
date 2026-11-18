CC = g++

CFLAGS = -O3 -fopenmp
INC    = -I/usr/include/hdf5/serial
# LIBDIR = -L/usr/lib/x86_64-linux-gnu
LIBS   = -lhdf5_serial

TARGET = m_exp
SRC = matrix_init.cpp matrix_utils.cpp main.cpp AliasTable.cpp m_exp.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
#	$(CC) $(CFLAGS) $(OBJ) $(LIBDIR) $(LIBS) -o $(TARGET)
	$(CC) $(CFLAGS) $(OBJ) $(LIBS) -o $(TARGET)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
