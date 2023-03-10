from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--long_term', action='store_true', default=False, help='test long term dataset')
        self.parser.add_argument('--results_dir', type=str, default='./results', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--max_test_num', type=int, default=-1, help='how many test images to run')
        self.parser.add_argument('--eval_frame', type=int, default=10, help='')
        self.isTrain = False
