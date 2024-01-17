import numpy as np
from tqdm import tqdm
from scipy import stats
from nltools.stats import _calc_pvalue as calc_pvalue

def get_permuted_pval(nsubj, nboot, all_scores, all_possible_zpairs, replacement=True, write_zstat=True, write_boot=False):
    n_centers = all_scores.shape[0]

    if write_boot:
        boot_z = np.zeros((n_centers, nsubj,nboot))
    
    if write_zstat:
        print('computing Z stats....')

    if n_centers > 1000:
        # we can't run all centers at once, that will take too much memory
        # so lets to some chunking
        chunked_center = np.split(np.arange(n_centers),
                                  np.linspace(0, n_centers,
                                              101, dtype=int)[1:-1])
        pvals = np.zeros((n_centers,))
        zstats = np.zeros((n_centers,))
        # loop over chunks
        for chunks in tqdm(chunked_center, desc='Calculating Permutations...'):
            zstat = []
            pval = []
            
            for c in chunks:
                # grab this center and neighbors
                voxel_score = all_scores[c]
                
                center_data = np.zeros((nsubj,nboot))
                for niter in range(nboot):        
                    np.random.seed(niter)
                    if write_boot:
                        boot_z[c, :, niter] = np.random.choice(all_possible_zpairs[c], nsubj, replace=replacement)
                    else:
                        center_data[:, niter] = np.random.choice(all_possible_zpairs[c], nsubj, replace=replacement)

                # compute p val for voxel
                if write_zstat:
                    this_zstat = (voxel_score - center_data.mean(axis=0).mean()) / center_data.mean(axis=0).std(axis=0)
                    this_z_hist =(center_data.mean(axis=0) - center_data.mean(axis=0).mean()) / center_data.mean(axis=0).std(axis=0)
                    zstat += [this_zstat]

                    this_pval = stats.norm.sf(abs(this_zstat))*2
                    pval += [this_pval]
                else:
                    if write_boot:
                        pval += [calc_pvalue(all_p=np.mean(boot_z[c, :, :], axis=0),
                                        stat=voxel_score, tail=2)]
                    else:
                        pval += [calc_pvalue(all_p=np.mean(center_data, axis=0),
                                        stat=voxel_score, tail=2)]
            if write_zstat:
                zstats[chunks] = zstat
            pvals[chunks] = pval

    if write_zstat:
        return zstats, pvals
    
    if write_boot:
        return boot_z, pvals
    else:
        return 0, pvals