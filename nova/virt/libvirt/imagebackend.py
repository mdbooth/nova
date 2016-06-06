# Copyright 2012 Grid Dynamics
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import abc
import base64
import contextlib
import functools
import os
import shutil

from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import fileutils
from oslo_utils import strutils
import six

import nova.conf
from nova import exception
from nova.i18n import _
from nova.i18n import _LE, _LI, _LW
from nova import image
from nova import keymgr
from nova import utils
from nova.virt.disk import api as disk
from nova.virt.image import model as imgmodel
from nova.virt import images
from nova.virt.libvirt import config as vconfig
from nova.virt.libvirt import imagecache
from nova.virt.libvirt.storage import dmcrypt
from nova.virt.libvirt.storage import lvm
from nova.virt.libvirt.storage import rbd_utils
from nova.virt.libvirt import utils as libvirt_utils

CONF = nova.conf.CONF

LOG = logging.getLogger(__name__)
IMAGE_API = image.API()


@six.add_metaclass(abc.ABCMeta)
class Image(object):

    SUPPORTS_CLONE = False

    def __init__(self, source_type, driver_format, is_block_dev=False):
        """Image initialization.

        :source_type: block or file
        :driver_format: raw or qcow2
        :is_block_dev:
        """
        if (CONF.ephemeral_storage_encryption.enabled and
                not self._supports_encryption()):
            raise exception.NovaException(_('Incompatible settings: '
                                  'ephemeral storage encryption is supported '
                                  'only for LVM images.'))

        self.source_type = source_type
        self.driver_format = driver_format
        self.driver_io = None
        self.discard_mode = CONF.libvirt.hw_disk_discard
        self.is_block_dev = is_block_dev
        self.preallocate = False

        # NOTE(dripton): We store lines of json (path, disk_format) in this
        # file, for some image types, to prevent attacks based on changing the
        # disk_format.
        self.disk_info_path = None

        # NOTE(mikal): We need a lock directory which is shared along with
        # instance files, to cover the scenario where multiple compute nodes
        # are trying to create a base file at the same time
        self.lock_path = os.path.join(CONF.instances_path, 'locks')

    def _supports_encryption(self):
        """Used to test that the backend supports encryption.
        Override in the subclass if backend supports encryption.
        """
        return False

    def create_image(self, prepare_template, base, size, *args, **kwargs):
        raise NotImplementedError()  # TODO(diana): delete method

    def check_backing_from_func(self, func, cache_name, fallback=None):
        """Creates missing backing file from func (if applicable)."""
        pass

    def check_backing_from_image(self, context, image_id, fallback=None):
        """Creates missing backing file from glance image (if applicable)."""
        pass

    @abc.abstractmethod
    def create_from_func(self, context, func, cache_name, size=None,
                         fallback=None):
        """Create a disk from the output of a function. Used to create
        ephemeral and swap disks.

        :context: The current request context.
        :func: A function, taking a path argument, which will write data
               to the given path.
        :cache_name: A name which can be used as a cache key for the output
                     of the given function.
        :size: The size of the disk to create, in bytes. This is a hint for
               efficiency. It is not used by all backends. If given, the caller
               must ensure that this corresponds exactly to the amout of data
               written by func().
        :fallback: A compute host to side-load the output from if it is not
                   already in the image cache.

        """
        pass

    @abc.abstractmethod
    def create_from_image(self, context, image_id, size=None, fallback=None):
        """Create a disk from a glance image.

        :context: The current request context.
        :image_id: The image_id of the image to be fetched.
        :size: The size of the disk to create, in bytes. If not given,
               the disk will be created with the same size as the source image.
        :fallback: A compute host to side-load the image from if it is no
                   longer in the image cache or in glance.

        """
        pass

    @abc.abstractmethod
    def resize_image(self, size):
        """Resize image to size (in bytes).

        :size: Desired size of image in bytes

        """
        pass

    def libvirt_info(self, disk_bus, disk_dev, device_type, cache_mode,
                     extra_specs, hypervisor_version):
        """Get `LibvirtConfigGuestDisk` filled for this image.

        :disk_dev: Disk bus device name
        :disk_bus: Disk bus type
        :device_type: Device type for this image.
        :cache_mode: Caching mode for this image
        :extra_specs: Instance type extra specs dict.
        :hypervisor_version: the hypervisor version
        """
        info = vconfig.LibvirtConfigGuestDisk()
        info.source_type = self.source_type
        info.source_device = device_type
        info.target_bus = disk_bus
        info.target_dev = disk_dev
        info.driver_cache = cache_mode
        info.driver_discard = self.discard_mode
        info.driver_io = self.driver_io
        info.driver_format = self.driver_format
        driver_name = libvirt_utils.pick_disk_driver_name(hypervisor_version,
                                                          self.is_block_dev)
        info.driver_name = driver_name
        info.source_path = self.path

        self.disk_qos(info, extra_specs)

        return info

    def disk_qos(self, info, extra_specs):
        tune_items = ['disk_read_bytes_sec', 'disk_read_iops_sec',
            'disk_write_bytes_sec', 'disk_write_iops_sec',
            'disk_total_bytes_sec', 'disk_total_iops_sec']
        for key, value in six.iteritems(extra_specs):
            scope = key.split(':')
            if len(scope) > 1 and scope[0] == 'quota':
                if scope[1] in tune_items:
                    setattr(info, scope[1], value)

    def libvirt_fs_info(self, target, driver_type=None):
        """Get `LibvirtConfigGuestFilesys` filled for this image.

        :target: target directory inside a container.
        :driver_type: filesystem driver type, can be loop
                      nbd or ploop.
        """
        info = vconfig.LibvirtConfigGuestFilesys()
        info.target_dir = target

        if self.is_block_dev:
            info.source_type = "block"
            info.source_dev = self.path
        else:
            info.source_type = "file"
            info.source_file = self.path
            info.driver_format = self.driver_format
            if driver_type:
                info.driver_type = driver_type
            else:
                if self.driver_format == "raw":
                    info.driver_type = "loop"
                else:
                    info.driver_type = "nbd"

        return info

    def exists(self):
        return os.path.exists(self.path)

    def cache(self, fetch_func, filename, size=None, *args, **kwargs):
        raise NotImplementedError()  # TODO(diana): delete method

    def _get_cached_output_path(self, func, cache_name, fallback):
        cache = imagecache.ImageCacheLocalDir.get()
        cache_path = cache.get_func_output_path(func, cache_name, fallback)
        return cache_path

    def _get_cached_image(self, context, image_id, size, fallback):
        cache = imagecache.ImageCacheLocalDir.get()
        image_info = cache.get_image_info(context, image_id, fallback)
        self.verify_base_size(None, size, image_info.virtual_size)
        return image_info

    def _resize_disk(self, size, virtual_size):
        if size and size > virtual_size:
            self.resize_image(size)

    def _preallocate_disk(self, size):
        if self.preallocate and self._can_fallocate():
            utils.execute('fallocate', '-n', '-l', size, self.path)

    def _can_fallocate(self):
        """Check once per class, whether fallocate(1) is available,
           and that the instances directory supports fallocate(2).
        """
        can_fallocate = getattr(self.__class__, 'can_fallocate', None)
        if can_fallocate is None:
            test_path = self.path + '.fallocate_test'
            _out, err = utils.trycmd('fallocate', '-l', '1', test_path)
            fileutils.delete_if_exists(test_path)
            can_fallocate = not err
            self.__class__.can_fallocate = can_fallocate
            if not can_fallocate:
                LOG.warning(_LW('Unable to preallocate image at path: '
                                '%(path)s'), {'path': self.path})
        return can_fallocate

    def verify_base_size(self, base, size, base_size=0):
        """Check that the base image is not larger than size.
           Since images can't be generally shrunk, enforce this
           constraint taking account of virtual image size.
        """

        # Note(pbrady): The size and min_disk parameters of a glance
        #  image are checked against the instance size before the image
        #  is even downloaded from glance, but currently min_disk is
        #  adjustable and doesn't currently account for virtual disk size,
        #  so we need this extra check here.
        # NOTE(cfb): Having a flavor that sets the root size to 0 and having
        #  nova effectively ignore that size and use the size of the
        #  image is considered a feature at this time, not a bug.

        if size is None:
            return

        if size and not base_size:
            base_size = self.get_disk_size(base)

        if size < base_size:
            msg = _LE('%(base)s virtual size %(base_size)s '
                      'larger than flavor root disk size %(size)s')
            LOG.error(msg % {'base': base,
                              'base_size': base_size,
                              'size': size})
            raise exception.FlavorDiskSmallerThanImage(
                flavor_size=size, image_size=base_size)

    def get_disk_size(self, name):
        return disk.get_disk_size(name)

    def snapshot_extract(self, target, out_format):
        raise NotImplementedError()

    def _get_driver_format(self):
        return self.driver_format

    def resolve_driver_format(self):
        """Return the driver format for self.path.

        First checks self.disk_info_path for an entry.
        If it's not there, calls self._get_driver_format(), and then
        stores the result in self.disk_info_path

        See https://bugs.launchpad.net/nova/+bug/1221190
        """
        def _dict_from_line(line):
            if not line:
                return {}
            try:
                return jsonutils.loads(line)
            except (TypeError, ValueError) as e:
                msg = (_("Could not load line %(line)s, got error "
                        "%(error)s") %
                        {'line': line, 'error': e})
                raise exception.InvalidDiskInfo(reason=msg)

        @utils.synchronized(self.disk_info_path, external=False,
                            lock_path=self.lock_path)
        def write_to_disk_info_file():
            # Use os.open to create it without group or world write permission.
            fd = os.open(self.disk_info_path, os.O_RDONLY | os.O_CREAT, 0o644)
            with os.fdopen(fd, "r") as disk_info_file:
                line = disk_info_file.read().rstrip()
                dct = _dict_from_line(line)

            if self.path in dct:
                msg = _("Attempted overwrite of an existing value.")
                raise exception.InvalidDiskInfo(reason=msg)
            dct.update({self.path: driver_format})

            tmp_path = self.disk_info_path + ".tmp"
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT, 0o644)
            with os.fdopen(fd, "w") as tmp_file:
                tmp_file.write('%s\n' % jsonutils.dumps(dct))
            os.rename(tmp_path, self.disk_info_path)

        try:
            if (self.disk_info_path is not None and
                        os.path.exists(self.disk_info_path)):
                with open(self.disk_info_path) as disk_info_file:
                    line = disk_info_file.read().rstrip()
                    dct = _dict_from_line(line)
                    for path, driver_format in six.iteritems(dct):
                        if path == self.path:
                            return driver_format
            driver_format = self._get_driver_format()
            if self.disk_info_path is not None:
                fileutils.ensure_tree(os.path.dirname(self.disk_info_path))
                write_to_disk_info_file()
        except OSError as e:
            raise exception.DiskInfoReadWriteFail(reason=six.text_type(e))
        return driver_format

    @staticmethod
    def is_shared_block_storage():
        """True if the backend puts images on a shared block storage."""
        return False

    @staticmethod
    def is_file_in_instance_path():
        """True if the backend stores images in files under instance path."""
        return False

    def clone(self, context, image_id_or_uri):
        raise NotImplementedError()  # TODO(diana): delete method

    def direct_snapshot(self, context, snapshot_name, image_format, image_id,
                        base_image_id):
        """Prepare a snapshot for direct reference from glance

        :raises: exception.ImageUnacceptable if it cannot be
                 referenced directly in the specified image format
        :returns: URL to be given to glance
        """
        raise NotImplementedError(_('direct_snapshot() is not implemented'))

    def cleanup_direct_snapshot(self, location, also_destroy_volume=False,
                                ignore_errors=False):
        """Performs any cleanup actions required after calling
        direct_snapshot(), for graceful exception handling and the like.

        This should be a no-op on any backend where it is not implemented.
        """
        pass

    def get_model(self, connection):
        """Get the image information model

        :returns: an instance of nova.virt.image.model.Image
        """
        raise NotImplementedError()

    def import_file(self, instance, local_file, remote_name):
        """Import an image from local storage into this backend.

        Import a local file into the store used by this image type. Note that
        this is a noop for stores using local disk (the local file is
        considered "in the store").

        If the image already exists it will be overridden by the new file

        :param local_file: path to the file to import
        :param remote_name: the name for the file in the store
        """

        # NOTE(mikal): this is a noop for now for all stores except RBD, but
        # we should talk about if we want this functionality for everything.
        pass

    def create_snap(self, name):
        """Create a snapshot on the image.  A noop on backends that don't
        support snapshots.

        :param name: name of the snapshot
        """
        pass

    def remove_snap(self, name, ignore_errors=False):
        """Remove a snapshot on the image.  A noop on backends that don't
        support snapshots.

        :param name: name of the snapshot
        :param ignore_errors: don't log errors if the snapshot does not exist
        """
        pass

    def rollback_to_snap(self, name):
        """Rollback the image to the named snapshot. A noop on backends that
        don't support snapshots.

        :param name: name of the snapshot
        """
        pass


class Flat(Image):
    """The Flat backend uses either raw or qcow2 storage. It never uses
    a backing store, so when using qcow2 it copies an image rather than
    creating an overlay. By default it creates raw files, but will use qcow2
    when creating a disk from a qcow2 if force_raw_images is not set in config.
    """
    def __init__(self, instance=None, disk_name=None, path=None):
        self.disk_name = disk_name
        super(Flat, self).__init__("file", "raw", is_block_dev=False)

        self.path = (path or
                     os.path.join(libvirt_utils.get_instance_path(instance),
                                  disk_name))
        self.preallocate = (
            strutils.to_slug(CONF.preallocate_images) == 'space')
        if self.preallocate:
            self.driver_io = "native"
        self.disk_info_path = os.path.join(os.path.dirname(self.path),
                                           'disk.info')
        self.correct_format()

    def _get_driver_format(self):
        try:
            data = images.qemu_img_info(self.path)
            return data.file_format
        except exception.InvalidDiskInfo as e:
            LOG.info(_LI('Failed to get image info from path %(path)s; '
                         'error: %(error)s'),
                      {'path': self.path,
                       'error': e})
            return 'raw'

    def _supports_encryption(self):
        # NOTE(dgenin): Kernel, ramdisk and disk.config are fetched using
        # the Flat backend regardless of which backend is configured for
        # ephemeral storage. Encryption for the Flat backend is not yet
        # implemented so this loophole is necessary to allow other
        # backends already supporting encryption to function. This can
        # be removed once encryption for Flat is implemented.
        if self.disk_name not in ['kernel', 'ramdisk', 'disk.config']:
            return False
        else:
            return True

    def correct_format(self):
        if os.path.exists(self.path):
            self.driver_format = self.resolve_driver_format()

    def create_from_func(self, context, func, cache_name, size=None,
                         fallback=None):
        cache_path = self._get_cached_output_path(func, cache_name, fallback)
        with self._create(size) as target:
            libvirt_utils.copy_image(cache_path, target)

    def create_from_image(self, context, image_id, size=None, fallback=None):
        image_info = self._get_cached_image(context, image_id, size, fallback)
        with self._create(size) as target:
            libvirt_utils.copy_image(image_info.path, target)
            self._resize_disk(size, image_info.virtual_size)

    @contextlib.contextmanager
    def _create(self, size):
        with fileutils.remove_path_on_error(self.path):
            yield self.path  # /instances/instance-uuid/some-disk
            self._preallocate_disk(size)
            self.correct_format()

    def resize_image(self, size):
        image = imgmodel.LocalFileImage(self.path, self.driver_format)
        disk.extend(image, size)

    def snapshot_extract(self, target, out_format):
        images.convert_image(self.path, target, self.driver_format, out_format)

    @staticmethod
    def is_file_in_instance_path():
        return True

    def get_model(self, connection):
        return imgmodel.LocalFileImage(self.path,
                                       imgmodel.FORMAT_RAW)


class Qcow2(Image):
    def __init__(self, instance=None, disk_name=None, path=None):
        super(Qcow2, self).__init__("file", "qcow2", is_block_dev=False)

        self.path = (path or
                     os.path.join(libvirt_utils.get_instance_path(instance),
                                  disk_name))
        self.preallocate = (
            strutils.to_slug(CONF.preallocate_images) == 'space')
        if self.preallocate:
            self.driver_io = "native"
        self.disk_info_path = os.path.join(os.path.dirname(self.path),
                                           'disk.info')
        self.resolve_driver_format()

    def check_backing_from_func(self, func, cache_name, fallback=None):
        cache = imagecache.ImageCacheLocalDir.get()
        missing_backing_file = self._check_backing_file()
        if missing_backing_file:
            cache.get_func_output_path(func, cache_name, fallback)

    def check_backing_from_image(self, context, image_id, fallback=None):
        cache = imagecache.ImageCacheLocalDir.get()
        missing_backing_file = self._check_backing_file()
        if missing_backing_file:
            cache.get_image_info(context, image_id, fallback)

    def _check_backing_file(self):
        backing_path = libvirt_utils.get_disk_backing_file(
            self.path, imgmodel.FORMAT_QCOW2)
        if backing_path is None or os.path.exists(backing_path):
            return  # doesn't use a backing file or it is already cached
        return backing_path

    def create_from_func(self, context, func, cache_name, size=None,
                         fallback=None):
        cache_path = self._get_cached_output_path(func, cache_name, fallback)
        with self._create(size) as target:
            libvirt_utils.create_cow_image(cache_path, target)

    def create_from_image(self, context, image_id, size=None, fallback=None):
        image_info = self._get_cached_image(context, image_id, size, fallback)
        with self._create(size) as target:
            libvirt_utils.create_cow_image(image_info.path, target)
            self._resize_disk(size, image_info.virtual_size)

    @contextlib.contextmanager
    def _create(self, size):
        with fileutils.remove_path_on_error(self.path):
            yield self.path  # /instances/instance-uuid/some-disk
            self._preallocate_disk(size)

    def resize_image(self, size):
        image = imgmodel.LocalFileImage(self.path, imgmodel.FORMAT_QCOW2)
        disk.extend(image, size)

    def snapshot_extract(self, target, out_format):
        libvirt_utils.extract_snapshot(self.path, 'qcow2',
                                       target,
                                       out_format)

    @staticmethod
    def is_file_in_instance_path():
        return True

    def get_model(self, connection):
        return imgmodel.LocalFileImage(self.path,
                                       imgmodel.FORMAT_QCOW2)


class Lvm(Image):
    @staticmethod
    def escape(filename):
        return filename.replace('_', '__')

    def __init__(self, instance=None, disk_name=None, path=None):
        super(Lvm, self).__init__("block", "raw", is_block_dev=True)

        self.ephemeral_key_uuid = instance.get('ephemeral_key_uuid')

        if self.ephemeral_key_uuid is not None:
            self.key_manager = keymgr.API(CONF)
        else:
            self.key_manager = None

        if path:
            self.path = path
            if self.ephemeral_key_uuid is None:
                info = lvm.volume_info(path)
                self.vg = info['VG']
                self.lv = info['LV']
            else:
                self.vg = CONF.libvirt.images_volume_group
        else:
            if not CONF.libvirt.images_volume_group:
                raise RuntimeError(_('You should specify'
                                     ' images_volume_group'
                                     ' flag to use LVM images.'))
            self.vg = CONF.libvirt.images_volume_group
            self.lv = '%s_%s' % (instance.uuid,
                                 self.escape(disk_name))
            if self.ephemeral_key_uuid is None:
                self.path = os.path.join('/dev', self.vg, self.lv)
            else:
                self.lv_path = os.path.join('/dev', self.vg, self.lv)
                self.path = '/dev/mapper/' + dmcrypt.volume_name(self.lv)

        # TODO(pbrady): possibly deprecate libvirt.sparse_logical_volumes
        # for the more general preallocate_images
        self.sparse = CONF.libvirt.sparse_logical_volumes
        self.preallocate = not self.sparse

        if not self.sparse:
            self.driver_io = "native"

    def _supports_encryption(self):
        return True

    def _can_fallocate(self):
        return False

    def create_from_func(self, context, func, cache_name, size=None,
                         fallback=None):
        cache_path = self._get_cached_output_path(func, cache_name, fallback)
        with self._create(context, size) as target:
            images.convert_image(cache_path, target,
                                 imgmodel.FORMAT_RAW, self.driver_format,
                                 run_as_root=True)

    def create_from_image(self, context, image_id, size=None, fallback=None):
        image_info = self._get_cached_image(context, image_id, size, fallback)
        with self._create(context, size) as target:
            images.convert_image(
                image_info.path, target, image_info.file_format,
                imgmodel.FORMAT_RAW, run_as_root=True)
            self._resize_disk(size, image_info.virtual_size)

    @contextlib.contextmanager
    def _create(self, context, size):
        with self.remove_volume_on_error(self.path):
            lvm.create_volume(self.vg, self.lv, size, sparse=self.sparse)
            if self.ephemeral_key_uuid is not None:
                self._encrypt_lvm_image(context)
            yield self.path  # /dev/some-volume-group/instance-uuid_some-disk

    def _get_encryption_key(self, context):
        try:
            return self.key_manager.get(
                context, self.ephemeral_key_uuid).get_encoded()
        except Exception:
            with excutils.save_and_reraise_exception():
                LOG.error(_LE("Failed to retrieve ephemeral encryption key"))

    def _encrypt_lvm_image(self, context):
        key = self._get_encryption_key(context)
        target = self.path.rpartition('/')[2]  # instance-uuid_some-disk
        dmcrypt.create_volume(
            target, self.lv_path, CONF.ephemeral_storage_encryption.cipher,
            CONF.ephemeral_storage_encryption.key_size, key)

    def resize_image(self, size):
        # size is ignored here, already taken into account by lvm.create_volume
        disk.resize2fs(self.path, run_as_root=True)

    @contextlib.contextmanager
    def remove_volume_on_error(self, path):
        try:
            yield
        except Exception:
            with excutils.save_and_reraise_exception():
                if self.ephemeral_key_uuid is None:
                    lvm.remove_volumes([path])
                else:
                    dmcrypt.delete_volume(path.rpartition('/')[2])
                    lvm.remove_volumes([self.lv_path])

    def snapshot_extract(self, target, out_format):
        images.convert_image(self.path, target, self.driver_format,
                             out_format, run_as_root=True)

    def get_model(self, connection):
        return imgmodel.LocalBlockImage(self.path)


class Rbd(Image):

    SUPPORTS_CLONE = True

    def __init__(self, instance=None, disk_name=None, path=None, **kwargs):
        super(Rbd, self).__init__("block", "rbd", is_block_dev=False)
        if path:
            try:
                self.rbd_name = path.split('/')[1]
            except IndexError:
                raise exception.InvalidDevicePath(path=path)
        else:
            self.rbd_name = '%s_%s' % (instance.uuid, disk_name)

        if not CONF.libvirt.images_rbd_pool:
            raise RuntimeError(_('You should specify'
                                 ' images_rbd_pool'
                                 ' flag to use rbd images.'))
        self.pool = CONF.libvirt.images_rbd_pool
        self.discard_mode = CONF.libvirt.hw_disk_discard
        self.rbd_user = CONF.libvirt.rbd_user
        self.ceph_conf = CONF.libvirt.images_rbd_ceph_conf

        self.driver = rbd_utils.RBDDriver(
            pool=self.pool,
            ceph_conf=self.ceph_conf,
            rbd_user=self.rbd_user)

        self.path = 'rbd:%s/%s' % (self.pool, self.rbd_name)
        if self.rbd_user:
            self.path += ':id=' + self.rbd_user
        if self.ceph_conf:
            self.path += ':conf=' + self.ceph_conf

    def libvirt_info(self, disk_bus, disk_dev, device_type, cache_mode,
            extra_specs, hypervisor_version):
        """Get `LibvirtConfigGuestDisk` filled for this image.

        :disk_dev: Disk bus device name
        :disk_bus: Disk bus type
        :device_type: Device type for this image.
        :cache_mode: Caching mode for this image
        :extra_specs: Instance type extra specs dict.
        """
        info = vconfig.LibvirtConfigGuestDisk()

        hosts, ports = self.driver.get_mon_addrs()
        info.source_device = device_type
        info.driver_format = 'raw'
        info.driver_cache = cache_mode
        info.driver_discard = self.discard_mode
        info.target_bus = disk_bus
        info.target_dev = disk_dev
        info.source_type = 'network'
        info.source_protocol = 'rbd'
        info.source_name = '%s/%s' % (self.pool, self.rbd_name)
        info.source_hosts = hosts
        info.source_ports = ports
        auth_enabled = (CONF.libvirt.rbd_user is not None)
        if CONF.libvirt.rbd_secret_uuid:
            info.auth_secret_uuid = CONF.libvirt.rbd_secret_uuid
            auth_enabled = True  # Force authentication locally
            if CONF.libvirt.rbd_user:
                info.auth_username = CONF.libvirt.rbd_user
        if auth_enabled:
            info.auth_secret_type = 'ceph'
            info.auth_secret_uuid = CONF.libvirt.rbd_secret_uuid

        self.disk_qos(info, extra_specs)

        return info

    def _can_fallocate(self):
        return False

    def exists(self):
        return self.driver.exists(self.rbd_name)

    def get_disk_size(self, name):
        """Returns the size of the virtual disk in bytes.

        The name argument is ignored since this backend already knows
        its name, and callers may pass a non-existent local file path.
        """
        return self.driver.size(self.rbd_name)

    def resize_image(self, size):
        self.driver.resize(self.rbd_name, size)

    def snapshot_extract(self, target, out_format):
        images.convert_image(self.path, target, 'raw', out_format)

    @staticmethod
    def is_shared_block_storage():
        return True

    def get_model(self, connection):
        secret = None
        if CONF.libvirt.rbd_secret_uuid:
            secretobj = connection.secretLookupByUUIDString(
                CONF.libvirt.rbd_secret_uuid)
            secret = base64.b64encode(secretobj.value())

        hosts, ports = self.driver.get_mon_addrs()
        servers = [str(':'.join(k)) for k in zip(hosts, ports)]

        return imgmodel.RBDImage(self.rbd_name,
                                 self.pool,
                                 self.rbd_user,
                                 secret,
                                 servers)

    def import_file(self, instance, local_file, remote_name):
        name = '%s_%s' % (instance.uuid, remote_name)
        if self.exists():
            self.driver.remove_image(name)
        self.driver.import_image(local_file, name)

    def create_from_func(self, context, func, cache_name, size=None,
                         fallback=None):
        cache_path = self._get_cached_output_path(func, cache_name, fallback)
        with self.remove_volume_on_error():
            self.driver.import_image(cache_path, self.rbd_name)

    def create_from_image(self, context, image_id, size=None, fallback=None):
        if self._clone_from_glance_location(context, image_id, size):
            return
        image_info = self._get_cached_image(context, image_id, size, fallback)
        with self.remove_volume_on_error():
            self.driver.import_image(image_info.path, self.rbd_name)
            self._resize_disk(size, image_info.virtual_size)

    def _clone_from_glance_location(self, context, image_id, size):
        image_meta = IMAGE_API.get(context, image_id, include_locations=True)
        locations = image_meta['locations']
        for location in locations:
            if self.driver.is_cloneable(location, image_meta):
                with self.remove_volume_on_error():
                    self.driver.clone(location, self.rbd_name)
                    # TODO(mdbooth): It would be better to verify
                    # base size before cloning the disk, but we'd
                    # need some additional methods in rbd_utils.
                    self.verify_base_size(self.rbd_name, size)
                    return location

    @contextlib.contextmanager
    def remove_volume_on_error(self):
        try:
            yield
        except Exception:
            with excutils.save_and_reraise_exception():
                if self.exists():
                    self.driver.remove_image(self.rbd_name)

    def create_snap(self, name):
        return self.driver.create_snap(self.rbd_name, name)

    def remove_snap(self, name, ignore_errors=False):
        return self.driver.remove_snap(self.rbd_name, name, ignore_errors)

    def rollback_to_snap(self, name):
        return self.driver.rollback_to_snap(self.rbd_name, name)

    def _get_parent_pool(self, context, base_image_id, fsid):
        parent_pool = None
        try:
            # The easy way -- the image is an RBD clone, so use the parent
            # images' storage pool
            parent_pool, _im, _snap = self.driver.parent_info(self.rbd_name)
        except exception.ImageUnacceptable:
            # The hard way -- the image is itself a parent, so ask Glance
            # where it came from
            LOG.debug('No parent info for %s; asking the Image API where its '
                      'store is', base_image_id)
            try:
                image_meta = IMAGE_API.get(context, base_image_id,
                                           include_locations=True)
            except Exception as e:
                LOG.debug('Unable to get image %(image_id)s; error: %(error)s',
                          {'image_id': base_image_id, 'error': e})
                image_meta = {}

            # Find the first location that is in the same RBD cluster
            for location in image_meta.get('locations', []):
                try:
                    parent_fsid, parent_pool, _im, _snap = \
                        self.driver.parse_url(location['url'])
                    if parent_fsid == fsid:
                        break
                    else:
                        parent_pool = None
                except exception.ImageUnacceptable:
                    continue

        if not parent_pool:
            raise exception.ImageUnacceptable(
                    _('Cannot determine the parent storage pool for %s; '
                      'cannot determine where to store images') %
                    base_image_id)

        return parent_pool

    def direct_snapshot(self, context, snapshot_name, image_format,
                        image_id, base_image_id):
        """Creates an RBD snapshot directly.
        """
        fsid = self.driver.get_fsid()
        # NOTE(nic): Nova has zero comprehension of how Glance's image store
        # is configured, but we can infer what storage pool Glance is using
        # by looking at the parent image.  If using authx, write access should
        # be enabled on that pool for the Nova user
        parent_pool = self._get_parent_pool(context, base_image_id, fsid)

        # Snapshot the disk and clone it into Glance's storage pool.  librbd
        # requires that snapshots be set to "protected" in order to clone them
        self.driver.create_snap(self.rbd_name, snapshot_name, protect=True)
        location = {'url': 'rbd://%(fsid)s/%(pool)s/%(image)s/%(snap)s' %
                           dict(fsid=fsid,
                                pool=self.pool,
                                image=self.rbd_name,
                                snap=snapshot_name)}
        try:
            self.driver.clone(location, image_id, dest_pool=parent_pool)
            # Flatten the image, which detaches it from the source snapshot
            self.driver.flatten(image_id, pool=parent_pool)
        finally:
            # all done with the source snapshot, clean it up
            self.cleanup_direct_snapshot(location)

        # Glance makes a protected snapshot called 'snap' on uploaded
        # images and hands it out, so we'll do that too.  The name of
        # the snapshot doesn't really matter, this just uses what the
        # glance-store rbd backend sets (which is not configurable).
        self.driver.create_snap(image_id, 'snap', pool=parent_pool,
                                protect=True)
        return ('rbd://%(fsid)s/%(pool)s/%(image)s/snap' %
                dict(fsid=fsid, pool=parent_pool, image=image_id))

    def cleanup_direct_snapshot(self, location, also_destroy_volume=False,
                                ignore_errors=False):
        """Unprotects and destroys the name snapshot.

        With also_destroy_volume=True, it will also cleanup/destroy the parent
        volume.  This is useful for cleaning up when the target volume fails
        to snapshot properly.
        """
        if location:
            _fsid, _pool, _im, _snap = self.driver.parse_url(location['url'])
            self.driver.remove_snap(_im, _snap, pool=_pool, force=True,
                                    ignore_errors=ignore_errors)
            if also_destroy_volume:
                self.driver.destroy_volume(_im, pool=_pool)


class Ploop(Image):
    def __init__(self, instance=None, disk_name=None, path=None):
        super(Ploop, self).__init__("file", "ploop", is_block_dev=False)

        self.path = (path or
                     os.path.join(libvirt_utils.get_instance_path(instance),
                                  disk_name))
        self.resolve_driver_format()

    def resize_image(self, size):
        dd_path = os.path.join(self.path, "DiskDescriptor.xml")
        utils.execute('ploop', 'grow', '-s', '%dK' % (size >> 10), dd_path,
                      run_as_root=True)

    def _restore_descriptor(self, path, pcs_format, image_path):
        utils.execute('ploop', 'restore-descriptor', '-f', pcs_format,
                      path, image_path)

    def create_from_func(self, context, func, cache_name, size=None,
                         fallback=None):
        cache_path = self._get_cached_output_path(func, cache_name, fallback)
        with self._create(pcs_format='raw') as target:
            libvirt_utils.copy_image(cache_path, target)

    def create_from_image(self, context, image_id, size=None, fallback=None):
        pcs_format = self._verify_pcs_format(context, image_id)
        image_info = self._get_cached_image(context, image_id, size, fallback)
        with self._create(pcs_format) as target:
            libvirt_utils.copy_image(image_info.path, target)
            self._resize_disk(size, image_info.virtual_size)

    @contextlib.contextmanager
    def _create(self, pcs_format):
        target = os.path.join(self.path, 'root.hds')
        remove_func = functools.partial(
            fileutils.delete_if_exists, remove=shutil.rmtree)
        with fileutils.remove_path_on_error(self.path, remove=remove_func):
            fileutils.ensure_tree(self.path)
            yield target  # /instances/instance-uuid/some-disk/root.hds
            self._restore_descriptor(self.path, pcs_format, target)

    def _verify_pcs_format(self, context, image_id):
        if CONF.force_raw_images:
            pcs_format = 'raw'
        else:
            image_meta = IMAGE_API.get(context, image_id)
            disk_format = image_meta.get('disk_format')
            if disk_format == 'ploop':
                pcs_format = 'expanded'
            elif disk_format == 'raw':
                pcs_format = 'raw'
            else:
                reason = _("PCS doesn't support images in %s format. You "
                    "should either set force_raw_images=True in config or "
                    "upload an image in ploop or raw format.") % disk_format
                raise exception.ImageUnacceptable(
                    image_id=image_id, reason=reason)
        return pcs_format

    def snapshot_extract(self, target, out_format):
        img_path = os.path.join(self.path, "root.hds")
        libvirt_utils.extract_snapshot(img_path,
                                       'parallels',
                                       target,
                                       out_format)


class Backend(object):
    def __init__(self, use_cow):
        self.BACKEND = {
            'raw': Flat,
            'flat': Flat,
            'qcow2': Qcow2,
            'lvm': Lvm,
            'rbd': Rbd,
            'ploop': Ploop,
            'default': Qcow2 if use_cow else Flat
        }

    def backend(self, image_type=None):
        if not image_type:
            image_type = CONF.libvirt.images_type
        image = self.BACKEND.get(image_type)
        if not image:
            raise RuntimeError(_('Unknown image_type=%s') % image_type)
        return image

    def image(self, instance, disk_name, image_type=None):
        """Constructs image for selected backend

        :instance: Instance name.
        :name: Image name.
        :image_type: Image type.
                     Optional, is CONF.libvirt.images_type by default.
        """
        backend = self.backend(image_type)
        return backend(instance=instance, disk_name=disk_name)

    def snapshot(self, instance, disk_path, image_type=None):
        """Returns snapshot for given image

        :path: path to image
        :image_type: type of image
        """
        backend = self.backend(image_type)
        return backend(instance=instance, path=disk_path)
