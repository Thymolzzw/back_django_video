import os

from pathlib import Path

from demo_django.asr.pyTranscriber.pytranscriber.control.ctr_autosub import Ctr_Autosub
from demo_django.asr.pyTranscriber.pytranscriber.util.srtparser import SRTParser
from demo_django.asr.pyTranscriber.pytranscriber.util.util import MyUtil


class Ctr_Main():

    signalProgress = ['finish', 100]

    def listenerProgress(self, string, percent):

        self.signalProgress.emit(string, percent)


    def listenerBExec(self, listfile, output_folder):

        if not MyUtil.is_internet_connected():
            print("Error! You need to have internet connection to use pyTranscriber!")
        else:
            #extracts the two letter lang_code from the string on language selection
            selectedLanguage = 'en-US - English (United States)'
            indexSpace = selectedLanguage.index(" ")
            langCode = selectedLanguage[:indexSpace]

            listFiles = []
            listFiles.append(str(listfile))
            print('this is listFiles -- ' + str(listFiles))

            outputFolder = output_folder
            print('this is output folder -- ' + outputFolder)

        # extract the filename without extension from the path
        base = os.path.basename(listfile)
        # [0] is filename, [1] is file extension
        fileName = os.path.splitext(base)[0]

         # the output file has same name as input file, located on output Folder
         # with extension .srt
        pathOutputFolder = Path(output_folder)
        outputFileSRT = pathOutputFolder / (fileName + ".srt")
        outputFileTXT = pathOutputFolder / (fileName + ".txt")

        sourceFile = listfile
        outputFiles = [outputFileSRT, outputFileTXT]
        outputFileSRT = outputFiles[0]
        outputFileTXT = outputFiles[1]




        #run autosub
        fOutput = Ctr_Autosub.generate_subtitles(source_path = sourceFile,
                                    output = outputFileSRT,
                                    src_language = langCode)
        #if nothing was returned
        if not fOutput:
            print("Error! Unable to generate subtitles for file " + sourceFile + ".")
        elif fOutput != -1:
            #if the operation was not canceled

            #updated the progress message
            pass

            #parses the .srt subtitle file and export text to .txt file
            SRTParser.extractTextFromSRT(str(outputFileSRT))

