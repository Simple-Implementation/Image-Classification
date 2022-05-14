import yaml 
import munch

class ConfigManager(object):
  
    '''
        Configuration 파일을 관리하기 위한 클래스입니다.
    '''

    def __init__(self, args):

        self.config_file = args.config_file
        self.cfg = self.load_yaml(args.config_file)
        self.cfg = munch.munchify(self.cfg)
        self.cfg.training_keyword = args.training_keyword
        self.cfg.program_param['wandb_key'] = args.wandb_key
        if args.resume:
            self.cfg.load_path = args.load_path
        

    def load_yaml(self,file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.full_load(f)

        return data