import torch

class CoresetSelection(object):
    @staticmethod
    def score_monotonic_selection(data_score, key, ratio, descending, class_balance):
        score = data_score[key]
        score_sorted_index = score.argsort(descending=descending)
        total_num = ratio * data_score['targets'].shape[0]

        if class_balance:
            print('Class balance mode.')
            all_index = torch.arange(data_score['targets'].shape[0])
            #Permutation
            targets_list = data_score['targets'][score_sorted_index]
            targets_unique = torch.unique(targets_list)
            for target in targets_unique:
                target_index_mask = (targets_list == target)
                targets_num = target_index_mask.sum()

            #Guarantee the class ratio doesn't change
            selected_index = []
            for target in targets_unique:
                target_index_mask = (targets_list == target)
                target_index = all_index[target_index_mask]
                target_coreset_num = targets_num * ratio
                selected_index = selected_index + list(target_index[:int(target_coreset_num)])
            selected_index = torch.tensor(selected_index)
            print(f'High priority {key}: {score[score_sorted_index[selected_index][:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[selected_index][-15:]]}')

            return score_sorted_index[selected_index]

        else:
            print(f'High priority {key}: {score[score_sorted_index[:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[-15:]]}')
            return score_sorted_index[:int(total_num)]

    @staticmethod
    def mislabel_mask(data_score, mis_key, mis_num, mis_descending, pruning_key, inplace=True, verbose=True):
        # mask out the mislabel data (hard samples)
        mis_score = data_score[mis_key]
        mis_score_sorted_index = mis_score.argsort(descending=mis_descending)
        hard_index = mis_score_sorted_index[:mis_num]
        if verbose:
            print(f'Bad data -> High priority {mis_key}: {data_score[mis_key][hard_index][:15]}')
            print(f'Prune {hard_index.shape[0]} samples.')

        easy_index = mis_score_sorted_index[mis_num:]
        if inplace:
            data_score[pruning_key] = data_score[pruning_key][easy_index]

        return data_score, easy_index


    @staticmethod
    def stratified_sampling(data_score, pruning_key, coreset_num, stratas=50, random_select=True):
        '''
        set coreset_num = -1 to use same number of stratas
        '''
        # print('Using stratified sampling...')
        if pruning_key is not None:
            score = data_score[pruning_key]
        else:
            score = torch.tensor(data_score)
        
        if coreset_num != -1:
            total_num = coreset_num
        else:
            total_num = stratas

        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001   # ensure maximum score value is also included in the last strata
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        # breakpoint()
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)

        strata_num = torch.tensor(strata_num)
        # breakpoint()

        def bin_allocate(num, bins):
            sorted_index = torch.argsort(bins)
            sort_bins = bins[sorted_index]

            num_bin = bins.shape[0]

            rest_exp_num = num
            # budgets = torch.zeros((num_bin,)).type(torch.int)
            budgets = []
            reverse_i = num_bin - 1
            for i in range(num_bin):
                rest_bins = num_bin - i
                avg = rest_exp_num // rest_bins
                cur_num = min(sort_bins[i].item(), avg)
                # print("avg", avg, "cur_num", cur_num)
                # budgets[i] = cur_num
                budgets.append(cur_num)
                rest_exp_num -= cur_num
            # breakpoint()

            rst = torch.zeros((num_bin,)).type(torch.int)
            rst[sorted_index] = torch.tensor(budgets).type(torch.int)

            return rst

        budgets = bin_allocate(total_num, strata_num)
        # breakpoint()

        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(score.shape[0])

        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]

            if random_select:
                rand_index = torch.randperm(pool.shape[0])
                selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]
            else:
                selected_index += [idx.item() for idx in pool[:budgets[i]]]

        return selected_index, None

    @staticmethod
    def random_selection(total_num, num):
        print('Random selection.')
        score_random_index = torch.randperm(total_num)

        return score_random_index[:int(num)]