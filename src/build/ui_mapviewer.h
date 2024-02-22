/********************************************************************************
** Form generated from reading UI file 'mapviewer.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAPVIEWER_H
#define UI_MAPVIEWER_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QStatusBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MapViewer
{
public:
    QAction *actionCapture;
    QWidget *centralwidget;
    QHBoxLayout *horizontalLayout;
    QStatusBar *statusbar;
    QMenuBar *menuBar;
    QMenu *menuScreenshot;

    void setupUi(QMainWindow *MapViewer)
    {
        if (MapViewer->objectName().isEmpty())
            MapViewer->setObjectName(QString::fromUtf8("MapViewer"));
        MapViewer->resize(800, 600);
        actionCapture = new QAction(MapViewer);
        actionCapture->setObjectName(QString::fromUtf8("actionCapture"));
        centralwidget = new QWidget(MapViewer);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        horizontalLayout = new QHBoxLayout(centralwidget);
        horizontalLayout->setContentsMargins(1, 1, 1, 1);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        MapViewer->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(MapViewer);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MapViewer->setStatusBar(statusbar);
        menuBar = new QMenuBar(MapViewer);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 800, 25));
        menuScreenshot = new QMenu(menuBar);
        menuScreenshot->setObjectName(QString::fromUtf8("menuScreenshot"));
        MapViewer->setMenuBar(menuBar);

        menuBar->addAction(menuScreenshot->menuAction());
        menuScreenshot->addAction(actionCapture);

        retranslateUi(MapViewer);

        QMetaObject::connectSlotsByName(MapViewer);
    } // setupUi

    void retranslateUi(QMainWindow *MapViewer)
    {
        MapViewer->setWindowTitle(QApplication::translate("MapViewer", "Map Viewer", 0, QApplication::UnicodeUTF8));
        actionCapture->setText(QApplication::translate("MapViewer", "Capture", 0, QApplication::UnicodeUTF8));
        menuScreenshot->setTitle(QApplication::translate("MapViewer", "Screen", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MapViewer: public Ui_MapViewer {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAPVIEWER_H
