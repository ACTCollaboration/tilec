from tilec import pipeline

print(pipeline.get_qids(exps=['act','planck'],daynight='daynight'))
print(pipeline.get_qids(exps=['act','planck'],daynight='night'))
print(pipeline.get_qids(exps=['planck'],daynight='daynight'))
print(pipeline.get_qids(exps=['planck'],daynight='night'))
print(pipeline.get_qids(exps=['act'],daynight='daynight'))
print(pipeline.get_qids(exps=['act'],daynight='night'))
