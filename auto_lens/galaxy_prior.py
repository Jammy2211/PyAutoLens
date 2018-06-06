import random
import string
from auto_lens import galaxy


class GalaxyPrior:
    def __init__(self, light_profile_classes=None, mass_profile_classes=None):
        self.id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

        self.light_profile_classes = light_profile_classes if light_profile_classes is not None else []
        self.mass_profile_classes = mass_profile_classes if mass_profile_classes is not None else []

    def attach_to_model_mapper(self, model_mapper):
        for num, light_profile_class in enumerate(self.light_profile_classes):
            model_mapper.add_class("{}_light_profile_{}".format(self.id, num), light_profile_class)

        for num, mass_profile_class in enumerate(self.mass_profile_classes):
            model_mapper.add_class("{}_mass_profile_{}".format(self.id, num), mass_profile_class)

    @property
    def light_profile_names(self):
        return ["{}_light_profile_{}".format(self.id, num) for num in range(len(self.light_profile_classes))]

    @property
    def mass_profile_names(self):
        return ["{}_mass_profile_{}".format(self.id, num) for num in range(len(self.mass_profile_classes))]

    def galaxy_for_model_instance(self, model_instance):
        light_profiles = []
        mass_profiles = []
        for num in range(len(self.light_profile_classes)):
            light_profiles.append(getattr(model_instance, "{}_light_profile_{}".format(self.id, num)))
        for num in range(len(self.mass_profile_classes)):
            mass_profiles.append(getattr(model_instance, "{}_mass_profile_{}".format(self.id, num)))

        return galaxy.Galaxy(light_profiles=light_profiles, mass_profiles=mass_profiles)
