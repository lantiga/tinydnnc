
FINAL_CFLAGS=$(CFLAGS)
FINAL_LDFLAGS=$(LDFLAGS)
FINAL_LDPATH=
TDNN_CC=gcc

DEPS_PATH=../deps

EXAMPLE_OBJ = example.o

INCLUDE_FLAGS=-I ../ -I $(DEPS_PATH)/tiny-dnn -I $(DEPS_PATH)/mnist

ifeq ($(uname_S),Linux)
	FINAL_CFLAGS += -fPIC -std=gnu99
	FINAL_LDFLAGS += -shared -Bsymbolic -lc
	FINAL_LDPATH += LD_LIBRARY_PATH=../build
else
	FINAL_CFLAGS += -dynamic -fno-common -std=c99
#	FINAL_LDFLAGS += -bundle -undefined dynamic_lookup -lc
	FINAL_LDFLAGS += -undefined dynamic_lookup -lc
	FINAL_LDPATH += DYLD_LIBRARY_PATH=../build
endif

all: example

example.o: example.c
	$(TDNN_CC) $(INCLUDE_FLAGS) -c $(FINAL_CFLAGS) example.c

example: example.o
	$(TDNN_CC) -L ../build $(FINAL_LDFLAGS) -ltinydnnc -o example $(EXAMPLE_OBJ)

clean:
	rm example.o example

