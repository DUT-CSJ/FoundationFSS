r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import open_clip


class DatasetFSS(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = split
        self.benchmark = 'fss'
        self.shot = shot

        self.base_path = os.path.join(datapath, 'FSS-1000')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        with open('./data/splits/fss/%s.txt' % split, 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()

        self.transform = transform
        self.class_names = ['ac wall', 'acorn', 'adhensive tape', 'adidas logo1', 'adidas logo2', 'afghan hound',
                            'african elephant', 'african grey', 'agama', 'air strip', 'aircraft carrier', 'airedale',
                            'airship', 'almond', 'ambulance', 'american staffordshire', 'anemone fish', 'angora',
                            'apple', 'armour', 'ashtray', 'assult rifle', 'aubergine', 'avocado',
                            'baboon', 'backpack', 'bagel', 'balance weight', 'bald eagle', 'ballpoint',
                            'banana', 'band-aid', 'banded gecko', 'barometer', 'baseball bat', 'baseball player',
                            'basketball', 'bassoon', 'bathtub', 'battery', 'beacon', 'beaker',
                            'beam bridge', 'bear', 'beaver', 'bedlington terrier', 'bee house', 'besom',
                            'birdhouse', 'bison', 'black grouse', 'black stork', 'black swan', 'blossom card',
                            'boa constrictor', 'bolotie', 'bomb', 'border terrier', 'boston bull', 'bottle cap',
                            'bouzouki', 'box turtle', 'bra', 'bracelet', 'bradypod', 'brain coral',
                            'brambling', 'brasscica', 'brick', 'brick card', 'brick tea', 'briefcase',
                            'brown bear', 'brush pen', 'buckingham palace', 'buckler', 'bullet train', 'bushtit',
                            'butterfly', 'cableways', 'cactus ball', 'cairn', 'camel', 'can opener',
                            'candle', 'cannon', 'canoe', 'capuchin', 'car mirror', 'car wheel',
                            'carbonara', 'carousel', 'carp', 'carrot', 'carton', 'cassette',
                            'cauliflower', 'celery', 'cello', 'chainsaw', 'chalk', 'cheese burger',
                            'chess bishop', 'chest', 'chickadee bird', 'chicken wings', 'chicory', 'chihuahua',
                            'children slide', 'chinese date', 'chopsticks', 'christmas stocking', 'cleaver', 'cn tower',
                            'cocacola', 'cocktail shaker', 'coffin', 'coho', 'collar', 'comb',
                            'computer mouse', 'conch', 'convertible', 'cornet', 'cosmetic brush', 'cottontail',
                            'coucal', 'cougar', 'cowboy hat', 'coyote', 'crane', 'crash helmet',
                            'cream', 'cristo redentor', 'croissant', 'cucumber', 'cumquat', 'dandie dinmont',
                            'dart', 'dhole', 'diamond', 'diaper', 'digital watch', 'dingo',
                            'dinosaur', 'donkey', 'dough', 'dragonfly', 'drake', 'drumstick',
                            'dugong', 'dumbbell', 'dutch oven', 'earplug', 'eft newt', 'egg tart',
                            'eggnog', 'egret', 'egyptian cat', 'electric fan', 'electronic toothbrush', 'eletrical switch',
                            'envelope', 'esport chair', 'espresso', 'excavator', 'face powder', 'feeder',
                            'ferret', 'fig', 'file cabinet', 'fire engine', 'flatworm', 'flowerpot',
                            'flute', 'flying disc', 'fork', 'forklift', 'fountain', 'frog',
                            'fur coat', 'garbage can', 'garbage truck', 'garfish', 'garlic', 'gas pump',
                            'gazelle', 'gecko', 'german pointer', 'giant panda', 'gliding lizard', 'globe',
                            'golden retriever', 'goldfish', 'golf ball', 'golfcart', 'goose', 'gorilla',
                            'gourd', 'grasshopper', 'great wall', 'green mamba', 'grey fox', 'grey whale',
                            'guacamole', 'guinea pig', 'gypsy moth', 'hamster', 'handshower', 'hard disk',
                            'hare', 'hartebeest', 'harvester', 'hawthorn', 'head cabbage', 'hen of the woods',
                            'hock', 'hook', 'hornbill', 'hornet', 'housefinch', 'howler monkey',
                            'hummingbird', 'hyena', 'ibex', 'igloo', 'indian cobra', 'indian elephant',
                            'jacamar', 'jackfruit', 'jacko lantern', 'jellyfish', 'jinrikisha', 'jordan logo',
                            'joystick', 'kangaroo', 'kappa logo', 'keyboard', 'killer whale', 'kinguin',
                            'kitchen knife', 'kite', 'koala', 'kremlin', 'ladder', 'ladle',
                            'lady slipper', 'ladybug', 'ladyfinger', 'lampshade', 'langur', 'lark',
                            'lawn mower', 'leatherback turtle', 'leeks', 'leopard', 'lesser panda', 'lhasa apso',
                            'lifeboat', 'light tube', 'lionfish', 'litchi', 'llama', 'loafer',
                            'lobster', 'lorikeet', 'lynx', 'macaque', 'mailbox', 'manx',
                            'maotai bottle', 'maraca', 'mario', 'marmot', 'marshmallow', 'mcdonald uncle',
                            'measuring cup', 'medical kit', 'meerkat', 'melon seed', 'memory stick', 'microphone',
                            'microwave', 'military vest', 'miniskirt', 'mink', 'modem', 'mongoose',
                            'monitor', 'mooli', 'mooncake', 'motarboard', 'motor scooter', 'mount fuji',
                            'mountain tent', 'mouse', 'mouthpiece', 'mud turtle', 'mule', 'muscle car',
                            'mushroom', 'nail scissor', 'neck brace', 'necklace', 'nematode', 'night snake',
                            'obelisk', 'ocicat', 'oil filter', 'okra', 'one-armed bandit', 'orange',
                            'ostrich', 'otter', 'owl', 'paddle', 'paint brush', 'panpipe',
                            'panther', 'papaya', 'paper crane', 'paper towel', 'parallel bars', 'park bench',
                            'patas', 'peacock', 'pen', 'pencil box', 'pencil sharpener2', 'pepitas',
                            'perfume', 'persian cat', 'persimmon', 'petri dish', 'pickup', 'pill bottle',
                            'pineapple', 'pingpong racket', 'pinwheel', 'pistachio', 'plate', 'poker',
                            'pokermon ball', 'polar bear', 'polecat', 'police car', 'pomelo', 'pool table',
                            'potato', 'potted plant', 'prairie chicken', 'prayer rug', 'printer', 'proboscis',
                            'psp', 'ptarmigan', 'pubg lvl3helmet', 'pufferfish', 'puma logo', 'punching bag',
                            'quad drone', 'quill pen', 'raccoon', 'radiator', 'radio', 'radio telescope',
                            'raft', 'rain barrel', 'recreational vehicle', 'red bayberry', 'red breasted merganser', 'red wolf',
                            'redheart', 'remote control', 'rhinoceros', 'ringlet butterfly', 'rock beauty', 'rocket',
                            'roller coaster', 'roller skate', 'rosehip', 'ruddy turnstone', 'ruffed grouse', 'running shoe',
                            'saluki', 'sandwich', 'sandwich cookies', 'sarong', 'scabbard', 'scorpion',
                            'screwdriver', 'scroll brush', 'seal', 'seatbelt', 'shakuhachi', 'shift gear',
                            'shih-tzu', 'shopping cart', 'shotgun', 'shower cap', 'sidewinder', 'single log',
                            'skua', 'skull', 'skunk', 'sleeping bag', 'sloth bear', 'snail',
                            'snake', 'snowball', 'snowmobile', 'soccer ball', 'solar dish', 'sombrero',
                            'space heater', 'space shuttle', 'spade', 'spark plug', 'sparrow', 'spatula',
                            'speaker', 'sponge', 'spoon', 'spoonbill', 'sports car', 'spotted salamander',
                            'spring scroll', 'squirrel', 'staffordshire', 'stapler', 'starfish', 'statue liberty',
                            'steak', 'steam locomotive', 'stinkhorn', 'stole', 'stool', 'stop sign',
                            'stove', 'strainer', 'streetcar', 'studio couch', 'stupa', 'submarine',
                            'sulphur crested', 'sundial', 'sunglasses', 'sunscreen', 'surfboard', 'swab',
                            'swimming glasses', 'swimming trunk', 'table lamp', 'tank', 'taxi', 'teapot',
                            'tebby cat', 'teddy', 'telescope', 'tennis racket', 'terrapin turtle', 'thatch',
                            'thimble', 'throne', 'tiger', 'tile roof', 'titi monkey', 'tobacco pipe',
                            'tofu', 'toilet plunger', 'toilet tissue', 'tokyo tower', 'tomb', 'toothbrush',
                            'toothpaste', 'torii', 'tractor', 'traffic light', 'trailer truck', 'trench coat',
                            'tresher', 'triceratops', 'trilobite', 'trimaran', 'trolleybus', 'turtle',
                            'usb', 'vending machine', 'vestment', 'victor icon', 'vine snake', 'violin',
                            'wafer', 'waffle', 'waffle iron', 'wall clock', 'wallaby', 'wallet',
                            'walnut', 'wardrobe', 'warthog', 'wash basin', 'washer', 'water bike',
                            'water polo', 'water snake', 'watermelon', 'whale', 'white shark', 'wild boar',
                            'window shade', 'witch hat', 'wok', 'wombat', 'wreck', 'wrench',
                            'yorkshire terrier', 'yurt', 'zebra', 'zucchini', 'ab wheel', 'abacus',
                            'ac ground', 'african crocodile', 'airliner', 'albatross', 'apron', 'arabian camel',
                            'arctic fox', 'armadillo', 'artichoke', 'baby', 'badger', 'balance beam',
                            'balloon', 'banjo', 'barbell', 'baseball', 'basset', 'beagle',
                            'bee', 'bee eater', 'beer bottle', 'beer glass', 'bell', 'bighorn sheep',
                            'bittern', 'blenheim spaniel', 'bluetick', 'bolete', 'bowtie', 'briard',
                            'broccoli', 'bulbul bird', 'cabbage', 'cactus', 'calculator', 'camomile',
                            'carambola', 'cardoon', 'ceiling fan', 'cheese', 'cheetah', 'cherry',
                            'chimpanzee', 'cigar', 'cigarette', 'coconut', 'coffeepot', 'colubus',
                            'common newt', 'consomme', 'conversion plug', 'conveyor', 'corn', 'cornmeal',
                            'cpu', 'crab', 'cradle', 'crayon', 'crepe', 'cricketball',
                            'crocodile', 'croquet ball', 'cup', 'cushion', 'daisy', 'digital clock',
                            'dishwasher', 'dowitcher', 'drum', 'eel', 'english setter', 'equestrian helmet',
                            'espresso maker', 'fire balloon', 'fire hydrant', 'fire screen', 'flat-coated retriever', 'folding chair',
                            'fox squirrel', 'french ball', 'frying pan', 'gas tank', 'giant schnauzer', 'gibbon',
                            'ginger', 'gyromitra', 'hair drier', 'hami melon', 'hammer', 'hammerhead shark',
                            'handcuff', 'handkerchief', 'har gow', 'harp', 'hatchet', 'hippo',
                            'hotdog', 'ice lolly', 'iceberg', 'icecream', 'impala', 'ipod',
                            'ironing board', 'jay bird', 'kazoo', 'kit fox', 'kobe logo', 'kwanyin',
                            'lacewing', 'laptop', 'lemon', 'lettuce', 'lion', 'lipstick',
                            'loggerhead turtle', 'loguat', 'louvre pyramid', 'lycaenid butterfly', 'macaw', 'mango',
                            'mashed potato', 'matchstick', 'mcdonald sign', 'microsd', 'monarch butterfly', 'monocycle',
                            'mortar', 'nagoya castle', 'narcissus', 'nike logo', 'olive', 'orang',
                            'oscilloscope', 'ox', 'panda', 'parachute', 'parking meter', 'partridge',
                            'pay phone', 'peanut', 'pear', 'piano keyboard', 'pickelhaube', 'pig',
                            'pillow', 'pinecone', 'pingpong ball', 'plaice', 'platypus', 'pomegranate',
                            'power drill', 'pretzel', 'projector', 'pubg airdrop', 'pubg lvl3backpack', 'pumpkin',
                            'rabbit', 'raven', 'razor', 'red fox', 'redshank', 'refrigerator',
                            'relay stick', 'revolver', 'rock snake', 'rose', 'saltshaker', 'sandal',
                            'sandbar', 'saxophone', 'scarerow', 'school bus', 'schooner', 'scissors',
                            'sea cucumber', 'sea urchin', 'sewing machine', 'shovel', 'shuriken', 'siamese cat',
                            'skateboard', 'ski mask', 'smoothing iron', 'sniper rifle', 'soap dispenser', 'sock',
                            'soup bowl', 'spider', 'spider monkey', 'squirrel monkey', 'stopwatch', 'stretcher',
                            'strongbox', 'sturgeon', 'suitcase', 'sungnyemun', 'sweatshirt', 'swim ring',
                            'syringe', 'tape player', 'taro', 'television', 'thor\'s hammer', 'three-toed sloth',
                            'thrush', 'tiger cat', 'tiger shark', 'timber wolf', 'toilet brush', 'toilet seat',
                            'tomato', 'totem pole', 'toucan', 'tow truck', 'tray', 'triumphal arch',
                            'tulip', 'turnstile', 'typewriter', 'umbrella', 'upright piano', 'vacuum',
                            'vase', 'volleyball', 'vulture', 'water ouzel', 'water tower', 'weasel',
                            'whiptail', 'whistle', 'white stork', 'windsor tie', 'wine bottle', 'wolf',
                            'woodpecker', 'yawl', 'yoga pad', 'yonex icon', 'abe\'s flyingfish', 'accordion',
                            'american alligator', 'american chamelon', 'andean condor', 'anise', 'apple icon', 'arch bridge',
                            'arrow', 'astronaut', 'australian terrier', 'bamboo dragonfly', 'bamboo slip', 'banana boat',
                            'barber shaver', 'bat', 'bath ball', 'beet root', 'bell pepper', 'big ben',
                            'black bear', 'bloodhound', 'boxing gloves', 'breast pump', 'broom', 'bucket',
                            'bulb', 'burj al', 'bus', 'bustard', 'cabbage butterfly', 'cablestayed bridge',
                            'cantilever bridge', 'canton tower', 'captain america shield', 'carriage', 'cathedrale paris', 'cd',
                            'chalk brush', 'chandelier', 'charge battery', 'chess king', 'chess knight', 'chess queen',
                            'chicken', 'chicken leg', 'chiffon cake', 'chinese knot', 'church', 'cicada',
                            'clam', 'clearwing flyingfish', 'cloud', 'coffee mug', 'coin', 'combination lock',
                            'condor', 'cricket', 'crt screen', 'cuckoo', 'curlew', 'dart target',
                            'delta wing', 'diver', 'doormat', 'doublebus', 'doughnut', 'downy pitch',
                            'drilling platform', 'dwarf beans', 'eagle', 'earphone1', 'earphone2', 'echidna',
                            'egg', 'electronic stove', 'english foxhound', 'f1 racing', 'fan', 'feather clothes',
                            'fennel bulb', 'ferrari911', 'fish', 'fish eagle', 'flamingo', 'fly',
                            'flying frog', 'flying geckos', 'flying snakes', 'flying squirrel', 'fox', 'french fries',
                            'ganeva chair', 'glider flyingfish', 'goblet', 'golden plover', 'goldfinch', 'groenendael',
                            'guitar', 'gym ball', 'haddock', 'hair razor', 'hang glider', 'harmonica',
                            'hawk', 'helicopter', 'hotel slipper', 'hover board', 'iguana', 'indri',
                            'ipad', 'iphone', 'iron man', 'jet aircraft', 'kart', 'key',
                            'knife', 'kunai', 'lapwing', 'leaf egg', 'leaf fan', 'leafhopper',
                            'leather shoes', 'leggings', 'lemur catta', 'letter opener', 'little blue heron', 'lotus',
                            'magpie bird', 'manatee', 'marimba', 'may bug', 'meatloaf', 'microscope',
                            'minicooper', 'missile', 'mite predator', 'mitten', 'moist proof pad', 'monkey',
                            'moon', 'motorbike', 'net surface shoes', 'nintendo 3ds', 'nintendo gba', 'nintendo sp',
                            'nintendo switch', 'nintendo wiiu', 'ocarina', 'oiltank car', 'onion', 'oriole',
                            'osprey', 'oyster', 'paper plane', 'parthenon', 'pencil sharpener1', 'peregine falcon',
                            'pheasant', 'phonograph', 'photocopier', 'pidan', 'pizza', 'plastic bag',
                            'poached egg', 'polo shirt', 'porcupine', 'potato chips', 'pspgo', 'pteropus',
                            'pumpkin pie', 'pyramid', 'pyramid cube', 'pyraminx', 'quail', 'quail egg',
                            'rally car', 'reel', 'reflex camera', 'rice cooker', 'rocking chair', 'rubber eraser',
                            'rubick cube', 'rugby ball', 'ruler', 'samarra mosque', 'santa sledge', 'screw',
                            'seagull', 'sealion', 'shower curtain', 'shumai', 'siamang', 'sled',
                            'snow leopard', 'snowman', 'snowplow', 'soap', 'soymilk machine', 'speedboat',
                            'spiderman', 'spinach', 'stealth aircraft', 'steering wheel', 'stingray', 'stone lion',
                            'stonechat', 'stork', 'strawberry', 'sulphur butterfly', 'sushi', 'swan',
                            'sydney opera house', 'taj mahal', 'tiltrotor', 'toaster', 'tower pisa', 'transport helicopter',
                            'tredmill', 'truss bridge', 'tunnel', 'twin tower', 'vacuum cup', 'villa savoye',
                            'vinyl', 'wagtail', 'wandering albatross', 'warehouse tray', 'warplane', 'wasp',
                            'water buffalo', 'water heater', 'wheelchair', 'whippet', 'white wolf', 'wig',
                            'windmill', 'window screen', 'wooden boat', 'wooden spoon']

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img, query_mask = self.transform(query_img, query_mask)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_transformed = [self.transform(support_img, support_cmask) for support_img, support_cmask in zip(support_imgs, support_masks)]
        support_masks = [x[1] for x in support_transformed]
        support_imgs = torch.stack([x[0] for x in support_transformed])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample),
                 'class_name': self.class_names[torch.tensor(class_sample)]}

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(os.path.dirname(query_name), query_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_sample = self.categories.index(query_name.split('/')[-2])
        if self.split == 'val':
            class_sample += 520
        elif self.split == 'test':
            class_sample += 760

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(range(1, 11), 1, replace=False)[0]
            support_name = os.path.join(os.path.dirname(query_name), str(support_name)) + '.jpg'
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata
