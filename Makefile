CPP = gcc
CFLAGS = -Wall -Wextra -Werror -O2 -g3 -std=c++0x -lstdc++ -fopenmp 
INC = -I ../eigen-eigen-1306d75b4a21
newton: src/newton-method.cpp
	$(CPP) $(CFLAGS) $(INC) src/newton-method.cpp ${LIBS} -o bin/newton-method

clean:
	rm bin/newton-method

test:
	bin/newton-method /scratch/02336/nrajani/SML/data/rcv1.tr /scratch/02336/nrajani/SML/data/rcv1.t

