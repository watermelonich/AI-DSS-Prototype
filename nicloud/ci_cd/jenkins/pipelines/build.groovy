// File: ci_cd/jenkins/pipelines/build.groovy

pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building the project...'
            }
        }
    }
}
