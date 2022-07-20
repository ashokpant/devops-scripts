cd /tmp/ && git clone https://github.com/openvenues/libpostal 
cd $HOME && \
	mkdir libpostal_data
cd /tmp/libpostal && \
	./bootstrap.sh && \
	./configure --datadir=$HOME/libpostal_data && \
	make -j4 && \
	sudo make install && \
	sudo	ldconfig

pip install postal
