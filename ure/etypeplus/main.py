import gc
import ure.scorer as scorer
import logging
import sys
import torch
import optuna
import ure.utils as utils
from ure.dataset import TSVDataset
from ure.hyperparams import parse_args
from ure.train_eval import load_vocabularies, train, test
from ure.etypeplus.encoder import Encoder
from ure.rel_dist import RelDist



def eval_etype(data):
    gold = []
    pred = []
    for item in data:
        pred.append(item["etype_pair"])
        gold.append(item["rel"])
        
    p, r, f1 = scorer.bcubed_score(gold, pred)
    print('b3: p={:.5f} r={:.5f} f1={:.5f}'.format(p, r, f1))
    homo, comp, v_m = scorer.v_measure(gold, pred)
    print('V-measure: hom.={:.5f} com.={:.5f} vm.={:.5f}'.format(homo, comp, v_m))
    ari = scorer.adjusted_rand_score(gold, pred)
    print('ARI={:.5f}'.format(ari))
    return f1


def objective(trial):
    # sample hyperparameters
    n_rels = trial.suggest_int("n_rels", 4, 12)
    ent_embdim = trial.suggest_categorical("ent_embdim", [20, 50, 100])
    k_samples = trial.suggest_categorical("k_samples", [3, 5, 10])

    alpha = trial.suggest_float("alpha", 0.05, 0.5, log=True)
    beta  = trial.suggest_float("beta",  0.01, 0.2, log=True)
    lr    = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    config_trial = config.copy()
    config_trial.update({
        "n_rels": n_rels,
        "ent_embdim": ent_embdim,
        "k_samples": k_samples,
        "loss_coef_alpha": alpha,
        "loss_coef_beta": beta,
        "lr": lr,
        "n_epochs": 5,        # IMPORTANT: short runs
        "patience": 2,
        "mode": "train"
    })

    model = RelDist(config={
            'n_rels': n_rels,
            'n_ents': vocas['entity'].size(),
            'ent_embdim': config["ent_embdim"],
            'n_etype_with_subjobj': vocas['etype_with_subjobj'].size(),
            'encoder_class': Encoder
        })
    
    model = utils.cuda(model)

    f1 = train(model, dataset, config)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return f1



if __name__ == '__main__':
    config = parse_args()
    print(config)
    vocas = load_vocabularies(config)

    # load dataset
    datadirs = {
        "train": config["train_path"],
        "dev": config["dev_path"],
        "test": config["test_path"]
    }

    k_samples = config["k_samples"]
    max_len = config["max_len"]
    freq_scale = config["freq_scale"]

    dataset = TSVDataset(
        datadirs, vocas=vocas,
        k_samples=k_samples, max_len=max_len,
        mask_entity_type=False)
    dataset.load(_format='txt')

    # create model
    print('create model')
    n_rels = config["n_rels"]
    print('N relations = {}'.format(n_rels))
    if config["mode"] == "etype":
        f1 = eval_etype(dataset.test)
    elif config["mode"] == 'train':
        model = RelDist(config={
            'n_rels': n_rels,
            'n_ents': vocas['entity'].size(),
            'ent_embdim': config["ent_embdim"],
            'n_etype_with_subjobj': vocas['etype_with_subjobj'].size(),
            'encoder_class': Encoder
        })
        model = utils.cuda(model)
        model.summary()
        train(model, dataset, config)
    elif config["mode"] == "tune":
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = "study-etypeplus-20250102-test-2"
        storage_name = "sqlite:///{}.db".format(study_name)

        study = optuna.create_study(study_name=study_name, storage=storage_name)
        study.optimize(objective, n_trials=1)
    elif config["mode"] == "test-best":
        
        study_name = config["study_name"]
        storage_name = config["storage_name"]
        study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
        
        config_eval={
            'n_rels': n_rels,
            'n_ents': vocas['entity'].size(),
            'ent_embdim': config["ent_embdim"],
            'n_etype_with_subjobj': vocas['etype_with_subjobj'].size(),
            'n_filters': config["n_filters"]
        }

        
        print(config_eval)
        config_eval.update(study.best_params)
        print(config_eval)

        model = Encoder(config=config_eval)
        model = utils.cuda(model)
        model.summary()
        model.eval()
        test(model, dataset, config)
    else:
        model = Encoder(config={
            'n_rels': n_rels,
            'n_ents': vocas['entity'].size(),
            'ent_embdim': config["ent_embdim"],
            'n_etype_with_subjobj': vocas['etype_with_subjobj'].size(),
            'n_filters': config["n_filters"]
        })
        model = utils.cuda(model)
        model.summary()
        model.eval()
        test(model, dataset, config)
