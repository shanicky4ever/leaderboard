from wandb import save
from team_code import image_agent
import torch
import torchvision
import numpy as np
import carla
import os
import cv2
import json

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))

def get_entry_point():
    return 'myImageAgent'

def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step):
    return image_agent.debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step)

def mkdir(step):
    os.mkdir(os.path.join('tmp_record', str(step*0.05)))

def save_info(step, data):
    with open(os.path.join('tmp_record', str(step*0.05), 'info.json'),'w') as f:
        json.dump(data, f, indent=4)

class myImageAgent(image_agent.ImageAgent):

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.save_attr=['gps','speed','compass','target']
        self.save_img = ['rgb','rgb_left','rgb_right']

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        points, (target_cam, _) = self.net.forward(img, target)
        points_cam = points.clone().cpu()
        points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
        points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
        points_cam = points_cam.squeeze()
        points_world = self.converter.cam_to_world(points_cam).numpy()

        aim = (points_world[1] + points_world[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        # desired_speed *= (1 - abs(angle)) ** 2

        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if self.step%20==0:
            mkdir(self.step)
            for im in self.save_img:
                cv2.imwrite(os.path.join('tmp_record', str(self.step*0.05), im+'.png'), tick_data[im])
            save_data = {}
            for attr in self.save_attr:
                save_data[attr] = tick_data[attr]
            save_data['steer'] = steer
            save_data['brake'] = 'True' if brake else 'False'
            save_data['throttle'] = throttle
            save_data['gps'] = save_data['gps'].tolist()
            save_data['target'] = save_data['target'].tolist()
            save_info(self.step, save_data)
            print(f'log success with step {self.step}')

        if DEBUG:
            debug_display(
                    tick_data, target_cam.squeeze(), points.cpu().squeeze(),
                    steer, throttle, brake, desired_speed,
                    self.step)

        return control