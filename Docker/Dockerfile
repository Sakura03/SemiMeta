from pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

# Add the keys and set permissions
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /root/.ssh && touch /root/.ssh/authorized_keys && \
    echo $ssh_pub_key > /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    mkdir -p /var/run/sshd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN echo "root:123" | chpasswd

RUN pip3 install tensorboardX && \
    apt-get install -y git

RUN git clone https://github.com/Sakura03/SemiMeta.git

ENTRYPOINT env | grep _ >> /etc/environment && service ssh start & /usr/sbin/sshd -D
