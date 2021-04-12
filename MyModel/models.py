# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Binner(models.Model):
    img = models.CharField(max_length=255, blank=True, null=True)
    video_id = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'binner'


class Country(models.Model):
    name = models.CharField(max_length=255)
    introduce = models.TextField(blank=True, null=True)
    is_delete = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'country'


class Operations(models.Model):
    id = models.IntegerField(primary_key=True)
    operation_type = models.CharField(max_length=255)
    user_id = models.IntegerField()
    video_id = models.IntegerField()
    comment = models.TextField(blank=True, null=True)
    operation_time = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'operations'


class People(models.Model):
    img = models.CharField(max_length=255, blank=True, null=True)
    introduce = models.TextField(blank=True, null=True)
    voice_feature_path = models.CharField(max_length=255, blank=True, null=True)
    name = models.CharField(max_length=255)
    is_delete = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'people'


class SourceInformation(models.Model):
    name = models.CharField(max_length=255)
    introduce = models.TextField(blank=True, null=True)
    source_url = models.CharField(max_length=255, blank=True, null=True)
    is_delete = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'source_information'


class Users(models.Model):
    name = models.CharField(max_length=255)
    account_name = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    type = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'users'


class Videos(models.Model):
    name = models.CharField(max_length=255)
    rel_path = models.CharField(max_length=255)
    introduce = models.TextField(blank=True, null=True)
    create_time = models.BigIntegerField(blank=True, null=True)
    subtitle = models.CharField(max_length=255, blank=True, null=True)
    source_id = models.IntegerField(blank=True, null=True)
    country_id = models.IntegerField(blank=True, null=True)
    tag = models.TextField(blank=True, null=True)
    snapshoot_img = models.CharField(max_length=255, blank=True, null=True)
    title = models.CharField(max_length=255, blank=True, null=True)
    text_location = models.CharField(max_length=255, blank=True, null=True)
    asr_path = models.CharField(max_length=255, blank=True, null=True)
    equipment_json_path = models.CharField(max_length=255, blank=True, null=True)
    ppt_pdf_path = models.CharField(max_length=255, blank=True, null=True)
    ppt_json_path = models.CharField(max_length=255, blank=True, null=True)
    face_npy_path = models.CharField(max_length=255, blank=True, null=True)
    create_user = models.IntegerField(blank=True, null=True)
    is_delete = models.IntegerField(blank=True, null=True)
    voice_json = models.CharField(max_length=255, blank=True, null=True)
    border_video_path = models.CharField(max_length=255, blank=True, null=True)
    translate_asr_path = models.CharField(max_length=255, blank=True, null=True)
    translate_subtitle = models.CharField(max_length=255, blank=True, null=True)
    text_pdf_location = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'videos'
