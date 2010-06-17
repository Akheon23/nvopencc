#
# Will compile each .i and test .ptx against .gold
# Note that a difference doesn't necessarily mean it is bad,
# may be an improvement.
#

# invoke make SIZE=64 for 64bit compiles, else defaults to 32
ifndef SIZE
SIZE=32
endif

ifeq ($(SIZE),32)
NVOPENCC = /home/mmurphy/sw/compiler/gpgpu/open64/src/targia32_nvisa/bin/nvopencc
else
NVOPENCC = /home/mmurphy/sw/compiler/gpgpu/open64/src/targia3264_nvisa/bin/nvopencc
endif

NVFLAGS =

include tests.mk

run : $(TESTS)

clean :
	rm -f *.log *.ptx* *.t *.B *.BI

%.log : %.i
	$(NVOPENCC) $(NVFLAGS) $<
	grep -v "^	//" $(SIZE)/$*.ptx > $*.ptx0
	grep -v "^	//" $*.ptx > $*.ptx1
	-diff $*.ptx0 $*.ptx1 > $@

