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
import collections
import contextlib
import functools
import os
import shutil

from oslo_concurrency import lockutils
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import fileutils
from oslo_utils import strutils
from oslo_utils import units
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

__imagebackend_opts = [
    cfg.StrOpt('images_type',
               default='default',
               choices=('raw', 'nobacking', 'qcow2', 'lvm', 'rbd', 'ploop',
                        'default'),
               help='VM Images format. If default is specified, then'
                    ' use_cow_images flag is used instead of this one.'),
    cfg.StrOpt('images_volume_group',
               help='LVM Volume Group that is used for VM images, when you'
                    ' specify images_type=lvm.'),
    cfg.BoolOpt('sparse_logical_volumes',
                default=False,
                help='Create sparse logical volumes (with virtualsize)'
                     ' if this flag is set to True.'),
    cfg.StrOpt('images_rbd_pool',
               default='rbd',
               help='The RADOS pool in which rbd volumes are stored'),
    cfg.StrOpt('images_rbd_ceph_conf',
               default='',  # default determined by librados
               help='Path to the ceph configuration file to use'),
    cfg.StrOpt('hw_disk_discard',
               choices=('ignore', 'unmap'),
               help='Discard option for nova managed disks. Need'
                    ' Libvirt(1.0.6) Qemu1.5 (raw format) Qemu1.6(qcow2'
                    ' format)'),
        ]

CONF = nova.conf.CONF
CONF.register_opts(__imagebackend_opts, 'libvirt')
CONF.import_opt('rbd_user', 'nova.virt.libvirt.volume.net', group='libvirt')
CONF.import_opt('rbd_secret_uuid', 'nova.virt.libvirt.volume.net',
                group='libvirt')

LOG = logging.getLogger(__name__)
IMAGE_API = image.API()


CachedImageInfo = collections.namedtuple(
    'CachedImageInfo', ('path', 'file_format', 'virtual_size', 'disk_size'))


class ImageCacheLocalPool(object):
    _singleton = None

    @classmethod
    def reset(cls):
        """Throw away the initialised singleton. Useful for testing."""
        cls._singleton = None

    @classmethod
    def get(cls):
        """Get the ImageCacheLocalPool singleton.

        Returns:
            ImageCacheLocalPool: ImageCacheLocalPool singleton
        """
        if cls._singleton is not None:
            return cls._singleton

        with lockutils.lock('libvirt.imagebackend.imagecache_local_pool'):
            if cls._singleton is None:
                cls._singleton = ImageCacheLocalPool()
            return cls._singleton

    def __init__(self):
        self.lock_path = os.path.join(CONF.instances_path, 'locks')
        self.base_dir = os.path.join(CONF.instances_path,
                                     CONF.image_cache_subdirectory_name)
        if not os.path.exists(self.base_dir):
            fileutils.ensure_tree(self.base_dir)

    def is_path_in_image_cache(self, path):
        return os.path.dirname(path) == self.base_dir

    @staticmethod
    def check_exists_and_mark_in_use(path):
        if os.path.exists(path):
            # NOTE(mikal): Update the mtime of the base file so the image
            # cache manager knows it is in use.
            libvirt_utils.update_mtime(path)
            return True

        return False

    def get_cached_image(self, context, image_id, instance,
                         fallback_from_host=None):
        """Fetch an image from the cache, side-loading it from
        `fallback_from_host` if it isn't in glance.

        Args:
            context: The current RequestContext.
            image_id: The image_id of the image to be fetched.
            instance: An instance we are fetching the image for. Used only
                    for user_id and project_id.
            fallback_from_host: A compute host to side-load the image from
                    if it is no longer in glance.

        Returns:
            CachedImageInfo: Metadata describing the cached image.
        """
        name = imagecache.get_cache_fname(image_id)
        path = os.path.join(self.base_dir, name)

        @utils.synchronized(name, external=True, lock_path=self.lock_path)
        def _sync():
            if self.check_exists_and_mark_in_use(path):
                return

            try:
                libvirt_utils.fetch_image(context, path, image_id,
                    instance.user_id, instance.project_id)
            except exception.ImageNotFound:
                if fallback_from_host is None:
                    raise

                LOG.debug("Image %(image_id)s doesn't exist anymore "
                          "on image service, attempting to copy "
                          "image from %(host)s",
                          {'image_id': image_id, 'host': fallback_from_host},
                          instance=instance)

                libvirt_utils.copy_image(src=path, dest=path,
                                         host=fallback_from_host,
                                         receive=True)

        _sync()

        # NOTE: We should persist this information somehow and return it
        # securely, because image inspection here isn't ideal. The image was
        # sanity checked in fetch_image when it was downloaded, so this
        # isn't the worst, but we could do better. Until then, this is
        # equivalent to what we have done before.
        info = images.qemu_img_info(path)
        return CachedImageInfo(path, info.file_format, info.virtual_size,
                               info.disk_size)

    def get_cached_func(self, func, name, fallback_from_host=None):
        """Get the cached output of a function from the cache. If it
        don't exist, try to side-load it from another compute host.
        Failing that, generate it again locally.
        Args:
            func: A function, taking a path as an argument, which will write
                data to path
            name: The name of the file in the cache.
            fallback_from_host: A compute host to side-load the output from.
        Returns:
            str: The path of the cached output.
        """
        path = os.path.join(self.base_dir, name)

        @utils.synchronized(name, external=True, lock_path=self.lock_path)
        def _sync():
            if self.check_exists_and_mark_in_use(path):
                return

            created = False
            if fallback_from_host is not None:
                try:
                    # Ideally we'll get the original backing file from the
                    # source host. The reason we prefer this is that if we
                    # have to generate a new one, or if the output already
                    # existed in our own cache from a previous run,
                    # for that matter, the output is guaranteed to be
                    # different. This is a bug which results in (hopefully
                    # only) subtle data corruption.
                    libvirt_utils.copy_image(src=path, dest=path,
                                             host=fallback_from_host,
                                             receive=True)
                    created = True
                except processutils.ProcessExecutionError:
                    LOG.exception(_LW('Failed to side-load %(path)s from '
                                      '%(host)s'),
                                  {'path': path, 'host': fallback_from_host})

            # Generate a new one as a last resort
            if not created:
                func(path)

        _sync()

        return path


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
        self.locked = False

    def _supports_encryption(self):
        """Used to test that the backend supports encryption.
        Override in the subclass if backend supports encryption.
        """
        return False

    @abc.abstractmethod
    def create_image(self, prepare_template, base, size, *args, **kwargs):
        """Create image from template.

        Contains specific behavior for each image type.

        :prepare_template: function, that creates template.
                           Should accept `target` argument.
        :base: Template name
        :size: Size of created image in bytes

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

    def preallocate_disk(self, size, path=None):
        if path is None:
            path = self.path

        utils.execute('fallocate', '-n', '-l', size, path)

    def create_from_func(self, context, func, cache_name, size,
                         fallback_from_host=None):
        """Create a disk from the output of a function. Used to create
        ephemeral and swap disks.

        Args:
            context: The current request context
            func: A create function which takes a path as its single
                argument, which will write the initial contents of the new
                disk.
            cache_name: A name which can be used as a key to cache the
                output of the function.
            size: The size of the disk to create, in bytes.
            fallback_from_host: If set and the image isn't cached, attempt to
                side-load it from the cache on this compute host.
        """
        raise NotImplementedError()

    def check_backing_from_func(self, func, cache_name,
                                fallback_from_host=None):
        pass

    def create_from_image(self, context, image_id, instance, size,
                          fallback_from_host=None):
        raise NotImplementedError()

    def check_backing_from_image(self, context, image_id, instance,
                                 fallback_from_host=None):
        pass

    def cache(self, fetch_func, filename, size=None, *args, **kwargs):
        """Creates image from template.

        Ensures that template and image not already exists.
        Ensures that base directory exists.
        Synchronizes on template fetching.

        :fetch_func: Function that creates the base image
                     Should accept `target` argument.
        :filename: Name of the file in the image directory
        :size: Size of created image in bytes (optional)
        """
        @utils.synchronized(filename, external=True, lock_path=self.lock_path)
        def fetch_func_sync(target, *args, **kwargs):
            # The image may have been fetched while a subsequent
            # call was waiting to obtain the lock.
            if not os.path.exists(target):
                fetch_func(target=target, *args, **kwargs)

        base_dir = os.path.join(CONF.instances_path,
                                CONF.image_cache_subdirectory_name)
        if not os.path.exists(base_dir):
            fileutils.ensure_tree(base_dir)
        base = os.path.join(base_dir, filename)

        if not self.exists() or not os.path.exists(base):
            self.create_image(fetch_func_sync, base, size,
                              *args, **kwargs)

        if size:
            if size > self.get_disk_size(base):
                self.resize_image(size)

            if (self.preallocate and self._can_fallocate() and
                    os.access(self.path, os.W_OK)):
                self.preallocate_disk(size)

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
        """Clone an image.

        Note that clone operation is backend-dependent. The backend may ask
        the image API for a list of image "locations" and select one or more
        of those locations to clone an image from.

        :param image_id_or_uri: The ID or URI of an image to clone.

        :raises: exception.ImageUnacceptable if it cannot be cloned
        """
        reason = _('clone() is not implemented')
        raise exception.ImageUnacceptable(image_id=image_id_or_uri,
                                          reason=reason)

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

    def _get_lock_name(self, base):
        """Get an image's name of a base file."""
        return os.path.split(base)[-1]

    def get_model(self, connection):
        """Get the image information model

        :returns: an instance of nova.virt.image.model.Image
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def lock(self):
        # We protect here against re-entrance. We assume that this object
        # isn't shared between multiple threads. The actual lock is
        # independent of this object, so multiple threads/processes is safe
        # as long as each has their own object.
        if self.locked:
            yield

        else:
            lock_name = os.path.basename(self.path)

            # This MUST be compatible with the lock used by
            # nova.utils.synchronized().
            # NOTE(mdbooth): Ideally there would be a way create a context
            # manager which uses nova.utils.synchronized directly.
            # Unfortunately I can't find one.
            with lockutils.lock(lock_name, lock_file_prefix='nova-',
                                external=True,
                                lock_path=self.lock_path):
                self.locked = True
                try:
                    yield
                finally:
                    self.locked = False

    def import_file(self, context, format, size=None):
        """Import from a local file.

        Yields a path which must be written to within the returned context.
        When the context exits, the contents of this path will be imported
        into the underlying store.

        If the image already exists it will be overwritten by the new file

        Args:
            context: A RequestContext
            format: The format of the data being imported
            size: The size of the data to be imported (in bytes)
        """
        # This should be abstract, but we don't have implementations for all
        # subclasses, yet (and they're not required, yet). This will be
        # fixed up in a later patch.
        raise NotImplementedError()

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


class NoBacking(Image):
    """The NoBacking backend uses either raw or qcow2 storage. It never uses
    a backing store, so when using qcow2 it copies an image rather than
    creating an overlay. By default it creates raw files, but will use qcow2
    when creating a disk from a qcow2 if force_raw_images is not set in config.
    """
    def __init__(self, instance=None, disk_name=None, path=None):
        self.disk_name = disk_name
        super(NoBacking, self).__init__("file", "raw", is_block_dev=False)

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
        # the NoBacking backend regardless of which backend is configured for
        # ephemeral storage. Encryption for the NoBacking backend is not yet
        # implemented so this loophole is necessary to allow other
        # backends already supporting encryption to function. This can
        # be removed once encryption for NoBacking is implemented.
        if self.disk_name not in ['kernel', 'ramdisk', 'disk.config']:
            return False
        else:
            return True

    def correct_format(self):
        if os.path.exists(self.path):
            self.driver_format = self.resolve_driver_format()

    @contextlib.contextmanager
    def import_file(self, context, format, size=0):
        # We haven't implemented format conversion when importing raw,
        # because nothing currently uses it.
        if format != imgmodel.FORMAT_RAW:
            raise NotImplementedError()

        @contextlib.contextmanager
        def _create_and_cleanup_on_error():
            with fileutils.remove_path_on_error(self.path):
                # Create a sparse output file of the correct size
                # Ideally we'd fallocate it here, but some operations will
                # truncate or overwrite the file, so we'd lose it. Instead,
                # we create a placeholder for operations which require it,
                # and fallocate afterwards.
                libvirt_utils.create_image(format, self.path, size)
                yield

        @contextlib.contextmanager
        def _no_cleanup():
            yield

        with self.lock():
            # Be sure we reuse any existing file, and don't delete an
            # existing file on error.
            cleanup = _no_cleanup \
                if self.exists() else _create_and_cleanup_on_error

            with cleanup():
                yield self.path

                if self.preallocate and self._can_fallocate():
                    self.preallocate_disk(size)

                # Write out the file format to disk.info
                self.correct_format()

    def create_from_func(self, context, func, cache_name, size,
                         fallback_from_host=None):
        cache = ImageCacheLocalPool.get()
        backing = cache.get_cached_func(func, cache_name,
                                        fallback_from_host=fallback_from_host)
        with self.import_file(context, imgmodel.FORMAT_RAW, size) as target:
            libvirt_utils.copy_image(backing, target)

    def create_from_image(self, context, image_id, instance, size,
                          fallback_from_host=None):
        # Get the image from the local image cache, downloading if necessary.
        imagecache_pool = ImageCacheLocalPool.get()
        cached_image = imagecache_pool.get_cached_image(
                context, image_id, instance,
                fallback_from_host=fallback_from_host)

        # Ensure that the disk is big enough for the image
        self.verify_base_size(None, size, cached_image.virtual_size)

        with self.import_file(context, imgmodel.FORMAT_RAW, size) as target:
            # Copy the cached image
            libvirt_utils.copy_image(cached_image.path, target)

            # Resize the disk if required
            if size > cached_image.virtual_size:
                image = imgmodel.LocalFileImage(target, self.driver_format)
                disk.extend(image, size)

    def create_image(self, prepare_template, base, size, *args, **kwargs):
        filename = self._get_lock_name(base)

        @utils.synchronized(filename, external=True, lock_path=self.lock_path)
        def copy_raw_image(base, target, size):
            libvirt_utils.copy_image(base, target)
            if size:
                image = imgmodel.LocalFileImage(target,
                                                self.driver_format)
                disk.extend(image, size)

        generating = 'image_id' not in kwargs
        if generating:
            if not self.exists():
                # Generating image in place
                prepare_template(target=self.path, *args, **kwargs)
        else:
            if not os.path.exists(base):
                prepare_template(target=base, max_size=size, *args, **kwargs)

            # NOTE(mikal): Update the mtime of the base file so the image
            # cache manager knows it is in use.
            libvirt_utils.update_mtime(base)
            self.verify_base_size(base, size)
            if not os.path.exists(self.path):
                with fileutils.remove_path_on_error(self.path):
                    copy_raw_image(base, self.path, size)

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

    def check_backing_from_func(self, func, cache_name,
                                fallback_from_host=None):
        backing_path = libvirt_utils.get_disk_backing_file(
            self.path, imgmodel.FORMAT_QCOW2)

        if backing_path is None:
            return

        cache_pool = ImageCacheLocalPool.get()
        if not cache_pool.is_path_in_image_cache(backing_path):
            raise exception.InvalidDiskFormat(
                message=_LE('Qcow2 disk has backing file which is not in the '
                            'image cache'))

        cache_name = os.path.basename(backing_path)
        cache_pool.get_cached_func(func, cache_name,
                                   fallback_from_host=fallback_from_host)

    def create_from_image(self, context, image_id, instance, size,
                          fallback_from_host=None):
        # Get the image from the local image cache, downloading if necessary.
        imagecache_pool = ImageCacheLocalPool.get()
        cached_image = imagecache_pool.get_cached_image(
                context, image_id, instance,
                fallback_from_host=fallback_from_host)

        # Ensure that the disk is big enough for the image
        self.verify_base_size(None, size, cached_image.virtual_size)

        with fileutils.remove_path_on_error(self.path):
            # TODO(pbrady): Consider copying the cow image here
            # with preallocation=metadata set for performance reasons.
            # This would be keyed on a 'preallocate_images' setting.

            # TODO(mdbooth): We have the backing file format here,
            # now. We should pass it explicitly.
            libvirt_utils.create_cow_image(cached_image.path, self.path)
            if size:
                image = imgmodel.LocalFileImage(self.path,
                                                imgmodel.FORMAT_QCOW2)
                disk.extend(image, size)

    def check_backing_from_image(self, context, image_id, instance,
                                 fallback_from_host=None):
        backing_path = libvirt_utils.get_disk_backing_file(self.path)
        if backing_path is None or os.path.exists(backing_path):
            return

        # Backing file is missing. Fetch the image.
        imagecache_pool = ImageCacheLocalPool.get()
        cached_image = imagecache_pool.get_cached_image(
                context, image_id, instance,
                fallback_from_host=fallback_from_host)

        # Check if that was it
        if cached_image.path == backing_path:
            return

        self._check_pre_grizzly(backing_path, cached_image)
        # NOTE(mdbooth): We don't raise an exception here. That doesn't seem
        # like a good idea.

    @staticmethod
    def _check_pre_grizzly(backing_path, cached_image):
        # Determine whether an existing qcow2 disk uses a legacy backing by
        # actually looking at the image itself and parsing the output of the
        # backing file it expects to be using.
        # See https://bugs.launchpad.net/nova/+bug/1185588

        # If this is a pre-Grizzly backing file, we expect the path to be:
        #   /path/to/cache/image_4
        backing_parts = backing_path.rpartition('_')

        # File prefix should be the cache image (/path/to/cache/image)
        if backing_parts[0] != cached_image.path:
            return

        # Suffix should be the image size (4)
        backing_size = backing_parts[-1]
        if not backing_size.isdigit():
            return

        # We have a legacy backing file
        backing_size = int(backing_size) * units.Gi

        with fileutils.remove_path_on_error(backing_path):
            libvirt_utils.copy_image(cached_image.path, backing_path)
            image = imgmodel.LocalFileImage(backing_path,
                                            imgmodel.FORMAT_QCOW2)
            disk.extend(image, backing_size)

    def create_image(self, prepare_template, base, size, *args, **kwargs):
        filename = self._get_lock_name(base)

        @utils.synchronized(filename, external=True, lock_path=self.lock_path)
        def copy_qcow2_image(base, target, size):
            # TODO(pbrady): Consider copying the cow image here
            # with preallocation=metadata set for performance reasons.
            # This would be keyed on a 'preallocate_images' setting.
            libvirt_utils.create_cow_image(base, target)
            if size:
                image = imgmodel.LocalFileImage(target, imgmodel.FORMAT_QCOW2)
                disk.extend(image, size)

        # Download the unmodified base image unless we already have a copy.
        if not os.path.exists(base):
            prepare_template(target=base, max_size=size, *args, **kwargs)

        # NOTE(ankit): Update the mtime of the base file so the image
        # cache manager knows it is in use.
        libvirt_utils.update_mtime(base)
        self.verify_base_size(base, size)

        legacy_backing_size = None
        legacy_base = base

        # Determine whether an existing qcow2 disk uses a legacy backing by
        # actually looking at the image itself and parsing the output of the
        # backing file it expects to be using.
        if os.path.exists(self.path):
            backing_path = libvirt_utils.get_disk_backing_file(self.path)
            if backing_path is not None:
                backing_file = os.path.basename(backing_path)
                backing_parts = backing_file.rpartition('_')
                if backing_file != backing_parts[-1] and \
                        backing_parts[-1].isdigit():
                    legacy_backing_size = int(backing_parts[-1])
                    legacy_base += '_%d' % legacy_backing_size
                    legacy_backing_size *= units.Gi

        # Create the legacy backing file if necessary.
        if legacy_backing_size:
            if not os.path.exists(legacy_base):
                with fileutils.remove_path_on_error(legacy_base):
                    libvirt_utils.copy_image(base, legacy_base)
                    image = imgmodel.LocalFileImage(legacy_base,
                                                    imgmodel.FORMAT_QCOW2)
                    disk.extend(image, legacy_backing_size)

        if not os.path.exists(self.path):
            with fileutils.remove_path_on_error(self.path):
                copy_qcow2_image(base, self.path, size)

    def resize_image(self, size):
        image = imgmodel.LocalFileImage(self.path, imgmodel.FORMAT_QCOW2)
        disk.extend(image, size)

    def create_from_func(self, context, func, cache_name, size,
                         fallback_from_host=None):
        # Qcow2 writes the function output to the cache, then creates an
        # overlay directly on the cached output.
        cache = ImageCacheLocalPool.get()
        backing = cache.get_cached_func(func, cache_name,
                                        fallback_from_host=fallback_from_host)

        with self.lock():
            libvirt_utils.create_cow_image(backing, self.path)

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

        self.ephemeral_key_uuid = instance.ephemeral_key_uuid

        if self.ephemeral_key_uuid is not None:
            self.key_manager = keymgr.API()
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

    @contextlib.contextmanager
    def import_file(self, context, format, size=0):
        @contextlib.contextmanager
        def _import_direct():
            # Import direct to LVM
            with self.remove_volume_on_error(self.path):
                lvm.create_volume(self.vg, self.lv, size, sparse=self.sparse)
                if self.ephemeral_key_uuid is not None:
                    try:
                        # NOTE(dgenin): Key manager corresponding to the
                        # specific backend catches and reraises an
                        # an exception if key retrieval fails.
                        key = self.key_manager.get_key(
                                context, self.ephemeral_key_uuid).get_encoded()
                    except Exception:
                        with excutils.save_and_reraise_exception():
                            LOG.error(_LE("Failed to retrieve ephemeral "
                                          "encryption key"))
                    dmcrypt.create_volume(
                            self.path.rpartition('/')[2], self.lv_path,
                            CONF.ephemeral_storage_encryption.cipher,
                            CONF.ephemeral_storage_encryption.key_size,
                            key)

                # NOTE: This is a wart, because the caller must know they need
                # to be root to write to it. It would be both cleaner and more
                # secure to chown this path to the nova user, and chown it back
                # again afterwards.
                yield self.path

        with self.lock():
            # If we're importing a raw file of known size, we can import direct
            if size is not None and format == imgmodel.FORMAT_RAW:
                with _import_direct() as path:
                    yield path

            # Otherwise we need to import to an intermediate temporary file and
            # convert the input
            else:
                with utils.tempdir() as tempdir:
                    tempfile = os.path.join(tempdir, 'lvm_import')
                    libvirt_utils.create_image(format, tempfile, size)
                    yield tempfile

                    with _import_direct() as path:
                        images.convert_image(tempfile, path, format,
                                             self.driver_format,
                                             run_as_root=True)

    def create_from_func(self, context, func, cache_name, size,
                         fallback_from_host=None):
        # Note that the Lvm backend doesn't use the image cache when creating
        # from a function
        with self.import_file(context, imgmodel.FORMAT_RAW, size) as target:
            func(target)

    def create_from_image(self, context, image_id, instance, size,
                          fallback_from_host=None):
        # Get the image from the local image cache, downloading if
        # necessary.
        imagecache_pool = ImageCacheLocalPool.get()
        cached_image = imagecache_pool.get_cached_image(
            context, image_id, instance,
            fallback_from_host=fallback_from_host)
        self.verify_base_size(None, size, cached_image.virtual_size)

        with self.import_file(context, cached_image.file_format, size) \
                as target:
            images.convert_image(cached_image.path, target,
                                 cached_image.file_format,
                                 self.driver_format,
                                 run_as_root=True)
            if size > cached_image.virtual_size:
                disk.resize2fs(target, run_as_root=True)

    def create_image(self, prepare_template, base, size, *args, **kwargs):
        def encrypt_lvm_image():
            dmcrypt.create_volume(self.path.rpartition('/')[2],
                                  self.lv_path,
                                  CONF.ephemeral_storage_encryption.cipher,
                                  CONF.ephemeral_storage_encryption.key_size,
                                  key)

        filename = self._get_lock_name(base)

        @utils.synchronized(filename, external=True, lock_path=self.lock_path)
        def create_lvm_image(base, size):
            base_size = disk.get_disk_size(base)
            self.verify_base_size(base, size, base_size=base_size)
            resize = size > base_size
            size = size if resize else base_size
            lvm.create_volume(self.vg, self.lv,
                                         size, sparse=self.sparse)
            if self.ephemeral_key_uuid is not None:
                encrypt_lvm_image()
            # NOTE: by calling convert_image_unsafe here we're
            # telling qemu-img convert to do format detection on the input,
            # because we don't know what the format is. For example,
            # we might have downloaded a qcow2 image, or created an
            # ephemeral filesystem locally, we just don't know here. Having
            # audited this, all current sources have been sanity checked,
            # either because they're locally generated, or because they have
            # come from images.fetch_to_raw. However, this is major code smell.
            images.convert_image_unsafe(base, self.path, self.driver_format,
                                        run_as_root=True)
            if resize:
                disk.resize2fs(self.path, run_as_root=True)

        generated = 'ephemeral_size' in kwargs
        if self.ephemeral_key_uuid is not None:
            if 'context' in kwargs:
                try:
                    # NOTE(dgenin): Key manager corresponding to the
                    # specific backend catches and reraises an
                    # an exception if key retrieval fails.
                    key = self.key_manager.get_key(kwargs['context'],
                            self.ephemeral_key_uuid).get_encoded()
                except Exception:
                    with excutils.save_and_reraise_exception():
                        LOG.error(_LE("Failed to retrieve ephemeral encryption"
                                      " key"))
            else:
                raise exception.NovaException(
                    _("Instance disk to be encrypted but no context provided"))
        # Generate images with specified size right on volume
        if generated and size:
            lvm.create_volume(self.vg, self.lv,
                                         size, sparse=self.sparse)
            with self.remove_volume_on_error(self.path):
                if self.ephemeral_key_uuid is not None:
                    encrypt_lvm_image()
                prepare_template(target=self.path, *args, **kwargs)
        else:
            if not os.path.exists(base):
                prepare_template(target=base, max_size=size, *args, **kwargs)
            with self.remove_volume_on_error(self.path):
                create_lvm_image(base, size)

    # NOTE(nic): Resizing the image is already handled in create_image(),
    # and migrate/resize is not supported with LVM yet, so this is a no-op
    def resize_image(self, size):
        pass

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

    @contextlib.contextmanager
    def remove_volume_on_error(self):
        try:
            yield
        except Exception:
            with excutils.save_and_reraise_exception(), self.lock():
                if self.exists():
                    self.driver.remove_image(self.rbd_name)

    def get_disk_size(self, name):
        """Returns the size of the virtual disk in bytes.

        The name argument is ignored since this backend already knows
        its name, and callers may pass a non-existent local file path.
        """
        return self.driver.size(self.rbd_name)

    def create_image(self, prepare_template, base, size, *args, **kwargs):

        if not self.exists():
            prepare_template(target=base, max_size=size, *args, **kwargs)

        # prepare_template() may have cloned the image into a new rbd
        # image already instead of downloading it locally
        if not self.exists():
            self.driver.import_image(base, self.rbd_name)
        self.verify_base_size(base, size)

        if size and size > self.get_disk_size(self.rbd_name):
            self.driver.resize(self.rbd_name, size)

    def resize_image(self, size):
        self.driver.resize(self.rbd_name, size)

    def snapshot_extract(self, target, out_format):
        images.convert_image(self.path, target, 'raw', out_format)

    @staticmethod
    def is_shared_block_storage():
        return True

    def clone(self, context, image_id_or_uri):
        image_meta = IMAGE_API.get(context, image_id_or_uri,
                                   include_locations=True)
        locations = image_meta['locations']

        LOG.debug('Image locations are: %(locs)s' % {'locs': locations})

        if image_meta.get('disk_format') not in ['raw', 'iso']:
            reason = _('Image is not raw format')
            raise exception.ImageUnacceptable(image_id=image_id_or_uri,
                                              reason=reason)

        for location in locations:
            if self.driver.is_cloneable(location, image_meta):
                return self.driver.clone(location, self.rbd_name)

        reason = _('No image locations are accessible')
        raise exception.ImageUnacceptable(image_id=image_id_or_uri,
                                          reason=reason)

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

    @contextlib.contextmanager
    def import_file(self, context, format, size=0):
        # NOTE: RBD supports locking, but this isn't yet exposed through
        # RBDDriver. We should do proper locking here. This mirrors current
        # behaviour, which uses a lock in the instance directory. This will
        # only work between hosts if there are shared instance directories.
        # We should subclass lock() for the Rbd backend to implement this.
        with self.lock():
            with utils.tempdir() as tempdir:
                import_file = os.path.join(tempdir, 'rbd_import')
                libvirt_utils.create_image(format, import_file, size)
                yield import_file

                if format != imgmodel.FORMAT_RAW:
                    converted_file = os.path.join(tempdir, 'rbd_converted')
                    images.convert_image(import_file, converted_file, format,
                                         imgmodel.FORMAT_RAW)
                    import_file = converted_file

                if self.exists():
                    self.driver.remove_image(self.rbd_name)
                self.driver.import_image(import_file, self.rbd_name)

    def create_from_func(self, context, func, cache_name, size,
                         fallback_from_host=None):
        # Rbd caches the function output to the local image cache,
        # then imports the output to rbd
        cache = ImageCacheLocalPool.get()
        cached = cache.get_cached_func(func, cache_name,
                                       fallback_from_host=fallback_from_host)
        with self.lock():
            self.driver.import_image(cached, self.rbd_name)

    def create_from_image(self, context, image_id, instance, size,
                          fallback_from_host=None):
        image_meta = IMAGE_API.get(context, image_id, include_locations=True)
        locations = image_meta['locations']

        # Look for a cloneable glance location
        for location in locations:
            if self.driver.is_cloneable(location, image_meta):
                with self.remove_volume_on_error():
                    with self.lock():
                        self.driver.clone(location, self.rbd_name)

                    # Check the size of the cloned disk.
                    # TODO(mdbooth): It would be better to check this before
                    # cloning the disk, but we'd need some additional
                    # methods in rbd_utils.
                    disk_size = self.driver.size(self.rbd_name)
                    self.verify_base_size(None, size, disk_size)
                    break

        # We didn't find a cloneable glance location, so download to the
        # local image cache and import it.
        else:
            imagecache_pool = ImageCacheLocalPool.get()
            cached_image = imagecache_pool.get_cached_image(
                context, image_id, instance,
                fallback_from_host=fallback_from_host)
            self.verify_base_size(None, size, cached_image.virtual_size)

            with self.lock():
                self.driver.import_image(cached_image.path, self.rbd_name)

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

    def create_image(self, prepare_template, base, size, *args, **kwargs):
        filename = os.path.split(base)[-1]

        @utils.synchronized(filename, external=True, lock_path=self.lock_path)
        def create_ploop_image(base, target, size):
            image_path = os.path.join(target, "root.hds")
            libvirt_utils.copy_image(base, image_path)
            self._restore_descriptor(target, self.pcs_format, image_path)
            if size:
                self.resize_image(size)

        if not os.path.exists(self.path):
            if CONF.force_raw_images:
                self.pcs_format = "raw"
            else:
                image_meta = IMAGE_API.get(kwargs["context"],
                                           kwargs["image_id"])
                format = image_meta.get("disk_format")
                if format == "ploop":
                    self.pcs_format = "expanded"
                elif format == "raw":
                    self.pcs_format = "raw"
                else:
                    reason = _("PCS doesn't support images in %s format."
                                " You should either set force_raw_images=True"
                                " in config or upload an image in ploop"
                                " or raw format.") % format
                    raise exception.ImageUnacceptable(
                                        image_id=kwargs["image_id"],
                                        reason=reason)

        if not os.path.exists(base):
            prepare_template(target=base, max_size=size, *args, **kwargs)
        self.verify_base_size(base, size)

        if os.path.exists(self.path):
            return

        fileutils.ensure_tree(self.path)

        remove_func = functools.partial(fileutils.delete_if_exists,
                                        remove=shutil.rmtree)
        with fileutils.remove_path_on_error(self.path, remove=remove_func):
            create_ploop_image(base, self.path, size)

    def resize_image(self, size):
        dd_path = os.path.join(self.path, "DiskDescriptor.xml")
        utils.execute('ploop', 'grow', '-s', '%dK' % (size >> 10), dd_path,
                      run_as_root=True)

    def _restore_descriptor(self, path, format, image_path):
        utils.execute('ploop', 'restore-descriptor', '-f', format,
                      path, image_path)

    @contextlib.contextmanager
    def import_file(self, context, format, size=None):
        if format not in (imgmodel.FORMAT_RAW, 'ploop'):
            raise NotImplementedError('Ploop can only import raw and ploop')

        with self.lock():
            remove_func = functools.partial(fileutils.delete_if_exists,
                                            remove=shutil.rmtree)
            with fileutils.remove_path_on_error(self.path, remove=remove_func):
                fileutils.ensure_tree(self.path)
                image_path = os.path.join(self.path, "root.hds")
                yield image_path
                self._restore_descriptor(self.path, format, image_path)
                if size is not None:
                    self.resize_image(size)

    def create_from_func(self, context, func, cache_name, size,
                         fallback_from_host=None):
        # Ploop caches the function output, then imports the cached output
        cache = ImageCacheLocalPool.get()
        cached = cache.get_cached_func(func, cache_name,
                                       fallback_from_host=fallback_from_host)

        with self.import_file(context, imgmodel.FORMAT_RAW, size) as \
                import_path:
            libvirt_utils.copy_image(cached, import_path)

    def create_from_image(self, context, image_id, instance, size,
                          fallback_from_host=None):
        if CONF.force_raw_images:
            pcs_format = "raw"
        else:
            image_meta = IMAGE_API.get(context, image_id)
            format = image_meta.get("disk_format")
            if format == "ploop":
                pcs_format = "expanded"
            elif format == "raw":
                pcs_format = "raw"
            else:
                reason = _("PCS doesn't support images in %s format."
                           " You should either set force_raw_images=True"
                           " in config or upload an image in ploop"
                           " or raw format.") % format
                raise exception.ImageUnacceptable(
                    image_id=image_id, reason=reason)

        imagecache_pool = ImageCacheLocalPool.get()
        cached_image = imagecache_pool.get_cached_image(
            context, image_id, instance,
            fallback_from_host=fallback_from_host)
        self.verify_base_size(None, size, cached_image.virtual_size)

        with self.import_file(context, pcs_format, size) as target:
            libvirt_utils.copy_image(cached_image.path, target)

    def snapshot_extract(self, target, out_format):
        img_path = os.path.join(self.path, "root.hds")
        libvirt_utils.extract_snapshot(img_path,
                                       'parallels',
                                       target,
                                       out_format)


class Backend(object):
    def __init__(self, use_cow):
        self.BACKEND = {
            'raw': NoBacking,
            'nobacking': NoBacking,
            'qcow2': Qcow2,
            'lvm': Lvm,
            'rbd': Rbd,
            'ploop': Ploop,
            'default': Qcow2 if use_cow else NoBacking
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
