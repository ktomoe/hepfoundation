import torch
import torch.nn as nn

from multiml import logger, const
from multiml.task.pytorch import PytorchBaseTask
from multiml.task.pytorch.samplers import SimpleBatchSampler


class MyTransformerTask(PytorchBaseTask):
    def set_hps(self, params):
        super().set_hps(params)

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

        self._model_args['task'] = self._data_id.split('_')[1]

        self._storegate.to_memory('features', phase='test')
        self._storegate.to_memory('labels', phase='test')
        self._storegate.to_memory('masks', phase='test')

        if isinstance(self._save_weights, str):
            self._save_weights += f'{self._data_id}.weight.{self._trial_id}'

        if isinstance(self._load_weights, str):

            if 'None' in self._load_weights:
                self._load_weights = False
                if 'per_params' in self._optimizer_args:
                    del self._optimizer_args['per_params'] # 0.01
            else:
                self._load_weights += f'.{self._trial_id}'
                self._load_weights += ':embedding:feature'

    def get_batch_sampler(self, phase, dataset):
        """Returns batch sampler."""
        sampler_args = dict(drop_last=False, batch_size=self._get_batch_size(phase))

        if phase in ('train', 'valid'):
            sampler = SimpleBatchSampler(
                len(dataset), batch_size=self._get_batch_size(phase), shuffle=True)
        else:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SequentialSampler(dataset), **sampler_args)
        return sampler

    @logger.logging
    def execute(self):
        """ Execute a task.
        """
        self.compile()
        self._storegate.set_mode('zarr')

        dataloaders = self.prepare_dataloaders()
        result = self.fit(dataloaders=dataloaders, dump=True)

        self._storegate.set_mode('numpy')
        pred = self.predict(dataloader=dataloaders['test'])
        self.update(data=pred, phase='test')
