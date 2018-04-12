"""
    This
"""

weights_proto = open('./test.txt', 'r')
deploy_proto = open('./deploy.prototxt', 'w')

for line in weights_proto.readlines():
    if 'data: ' not in line:
        deploy_proto.writelines(line)

